#include <cstdint>
#include <string>
#include <stdint.h>
#include <immintrin.h>
#include <vector>
#include "sdc_impl.h"
#include <pmmintrin.h>
#include <immintrin.h>


const int8_t centroids[16 * 16] = {
   112,97,82,67,52,37,22,7,-7,-22,-37,-52,-67,-82,-97,-112,
97,84,71,58,45,32,19,6,-6,-19,-32,-45,-58,-71,-84,-97,
82,71,60,49,38,27,16,5,-5,-16,-27,-38,-49,-60,-71,-82,
67,58,49,40,31,22,13,4,-4,-13,-22,-31,-40,-49,-58,-67,
52,45,38,31,24,17,10,3,-3,-10,-17,-24,-31,-38,-45,-52,
37,32,27,22,17,12,7,2,-2,-7,-12,-17,-22,-27,-32,-37,
22,19,16,13,10,7,4,1,-1,-4,-7,-10,-13,-16,-19,-22,
7,6,5,4,3,2,1,0,0,-1,-2,-3,-4,-5,-6,-7,
-7,-6,-5,-4,-3,-2,-1,0,0,1,2,3,4,5,6,7,
-22,-19,-16,-13,-10,-7,-4,-1,1,4,7,10,13,16,19,22,
-37,-32,-27,-22,-17,-12,-7,-2,2,7,12,17,22,27,32,37,
-52,-45,-38,-31,-24,-17,-10,-3,3,10,17,24,31,38,45,52,
-67,-58,-49,-40,-31,-22,-13,-4,4,13,22,31,40,49,58,67,
-82,-71,-60,-49,-38,-27,-16,-5,5,16,27,38,49,60,71,82,
-97,-84,-71,-58,-45,-32,-19,-6,6,19,32,45,58,71,84,97,
-112,-97,-82,-67,-52,-37,-22,-7,7,22,37,52,67,82,97,112,
};


inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

inline __m256i rbe_combine2x2(__m256i a, __m256i b, __m256i c) {
    __m256i a1b0 = _mm256_permute2f128_si256(a, b, 0x21);
    __m256i a0b1 = _mm256_blend_epi32(a, b, 0xF0);
    __m256i ab = _mm256_add_epi16(a1b0,a0b1);
    // return simd16uint16(ab);
    // __m256i sub_ab = _mm256_set1_epi16(sub_constx[0]);
    // __m256i sub_ab = _mm256_set1_epi16(408);
    // ab = _mm256_sub_epi16(ab, sub_ab);
    __m256i lo = _mm256_mullo_epi16(ab, c);
    
    return lo;
}


template <int NQ>
void rbe_kernel_scan(
        int nsq,
        const uint8_t* codes,
        const uint16_t* norms,
        const uint8_t* LUT) {
     uint16_t* res;
    __m256i accu[NQ][4];

    for (int q = 0; q < NQ; q++) {
        for (int b = 0; b < 4; b++) {
            accu[q][b] = _mm256_setzero_si256();
        }
    }

    for (int sq = 0; sq < nsq; sq += 2) {
        // prefetch

        __m256i c = _mm256_load_si256((__m256i const*)codes);
        __m256i mask = _mm256_set1_epi8(0xf);
        __m256i chi = _mm256_srli_epi16(c,4);
        chi = _mm256_and_si256(chi, mask);
        __m256i clo = _mm256_and_si256(c,mask);

        for (int q = 0; q < NQ; q++) {
            // load LUTs for 2 quantizers
            __m256i lut = _mm256_load_si256((__m256i const*)LUT);
            LUT += 32;

            __m256i res0 = _mm256_shuffle_epi8(lut, clo);
            __m256i res1 = _mm256_shuffle_epi8(lut, chi);

            accu[q][0] += res0;
            accu[q][1] += _mm256_srli_epi16(res0,8);
            accu[q][2] += res1;
            accu[q][3] += _mm256_srli_epi16(res1,8);
        }
    }

    __m256i temp_norm0 = _mm256_load_si256((__m256i const*)norms);
    norms +=16;
    __m256i temp_norm1 = _mm256_load_si256((__m256i const*)norms);

    for (int q = 0; q < NQ; q++) {

        accu[q][0] -= _mm256_slli_epi16(accu[q][1], 8 );
        __m256i dis_norm0 = rbe_combine2x2(accu[q][0], accu[q][1], temp_norm0);

        accu[q][2] -= _mm256_slli_epi16(accu[q][3], 8 );
        __m256i dis_norm1 = rbe_combine2x2(accu[q][2], accu[q][3], temp_norm1);
    
        _mm256_storeu_si256((__m256i*)res, dis_norm0);
        _mm256_storeu_si256((__m256i*)(res+16), dis_norm1);


    }
}


void IndexRbeScan::convert(int n, int bit_num, const uint8_t* x, std::vector<uint8_t>& codes){
 int uint8_nums = bit_num/8; 
 for(int i=0;  i< uint8_nums*n; i++ ){
    uint8_t first_code = (x[i] & 240)>>4;
    uint8_t second_code = x[i] >> 4;
    codes.push_back(first_code);
    codes.push_back(second_code);
 }
 return ;

}

IndexRbeScan::IndexRbeScan(int nbits) {
    return ;
}


int IndexRbeScan::add(int n,int bit_num, const uint8_t* x) {
    int code4_nums = bit_num/4; 
    int code8_nums = bit_num/8;
    int ntotal = roundup(n,32);
    int col_num = 32;
    int row_num = (ntotal/32) * M2 /2;
    int lack_num = ntotal - n;

    std::vector<uint8_t> temp_x;

    for(int i=0; i<n; i++){
        for(int j=0; j<code8_nums+1; j++){
            temp_x.push_back(x[i*(code8_nums+1)+j]);
        }
    }
    
    for (int i=0; i<lack_num; i++){
        for (int j=0; j<code8_nums+1; j++ ){
            temp_x.push_back(0);
        }
    }


    for (int i=0; i<row_num ; i++){
        for(int j=0; j < col_num; j++ ){
            codes.push_back( temp_x[ j*(code8_nums+1) + i ]);
            norms.push_back( (uint16_t) temp_x[ (j+1) * (code8_nums+1)]);
        }
    }
    return 0;
        
}

void IndexRbeScan::search(int n, const uint8_t *x, int bit_num, int k)
{
    const int QBS = 1;
    int code4_nums = bit_num / 4; 
    int code8_nums = bit_num / 8;
    int nsq = bit_num / 4;  // 256/4 =32
    constexpr int Q1 = QBS & 15;

    std::vector<uint8_t> lut_index_code;
    convert(n,bit_num,x,lut_index_code);
    for(int i=0; i < code4_nums*n; i++){
        for(int j =0; j<16; j++){
            lookuptable.push_back(centroids[lut_index_code[i] + j]);
        }
    }

    std::vector<uint16_t> res;
    const uint16_t* Norm = &norms[0];
    const uint8_t* Code = &codes[0];
    const uint8_t* LUT = &lookuptable[0];

    for(int64_t j0 = 0; j0 < ntotal; j0 += 32) {
        // uint16_t* Res = &res[0];

        // const uint8_t *Norm = norms.data();
        // const uint8_t *Code = codes.data();
        // const uint8_t *LUT = lookuptable.data();
        rbe_kernel_scan<Q1>(nsq, Code, Norm, LUT);
        LUT += Q1 * nsq * 16;
        Code += 32 * nsq / 2;
        Norm += 32;
    }
    return ;
}

int main(){
    IndexRbeScan idx_rbe(256);
    return 0;
}


