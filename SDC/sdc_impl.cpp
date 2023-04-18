#include <cstdint>
#include <string>
#include <stdint.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <vector>
#include <iostream>
#include "sdc_impl.h"
#include "utils.h"
const uint8_t centroids[16 * 16] = {
18,15,12,9,15,12,9,6,12,9,6,3,9,6,3,0,15,14,13,12,12,11,10,9,9,8,7,6,6,5,4,3,12,13,14,15,9,10,11,12,6,7,8,9,3,4,5,6,9,12,15,18,6,9,12,15,3,6,9,12,0,3,6,9,15,12,9,6,14,11,8,5,13,10,7,4,12,9,6,3,12,11,10,9,11,10,9,8,10,9,8,7,9,8,7,6,9,10,11,12,8,9,10,11,7,8,9,10,6,7,8,9,6,9,12,15,5,8,11,14,4,7,10,13,3,6,9,12,12,9,6,3,13,10,7,4,14,11,8,5,15,12,9,6,9,8,7,6,10,9,8,7,11,10,9,8,12,11,10,9,6,7,8,9,7,8,9,10,8,9,10,11,9,10,11,12,3,6,9,12,4,7,10,13,5,8,11,14,6,9,12,15,9,6,3,0,12,9,6,3,15,12,9,6,18,15,12,9,6,5,4,3,9,8,7,6,12,11,10,9,15,14,13,12,3,4,5,6,6,7,8,9,9,10,11,12,12,13,14,15,0,3,6,9,3,6,9,12,6,9,12,15,9,12,15,18
}
;
inline size_t roundup(size_t a, size_t b) {
	return (a + b - 1) / b * b;
}
inline __m256i rbe_combine2x2(__m256i a, __m256i b, __m256i c) {
	__m256i a1b0 = _mm256_permute2f128_si256(a, b, 0x21);
	__m256i a0b1 = _mm256_blend_epi32(a, b, 0xF0);
	__m256i ab = _mm256_add_epi16(a1b0,a0b1);

	// __m256i sub_ab = _mm256_set1_epi16(576);
	// ab = _mm256_sub_epi16(ab, sub_ab);
	__m256i lo = _mm256_mullo_epi16(ab, c);
	return lo;
}
template <class T>
static inline __m256i loadu_si256i(T *ptr) {
	// return _mm256_loadu_si256((__m256i *)ptr);
	return _mm256_lddqu_si256((__m256i *)ptr);
}
template <int Number_Query>
void rbe_kernel_scan(
        int number_sq,
        const uint8_t* codes,
        const uint16_t* norms,
        const uint8_t* LUT, uint16_t* res) {
	printf("[kernel] start scan\n");
	__m256i accu[Number_Query][4];
	for (int q = 0; q < Number_Query; q++) {
		for (int b = 0; b < 4; b++) {
			accu[q][b] = _mm256_setzero_si256();
		}
	}
	printf("[kernel] finish assign accu\n");
	for (int idx_sq = 0; idx_sq < number_sq; idx_sq += 2) {
		// printf("%u\n", (unsigned int)codes[0]);
		// __m256i c = _mm256_load_si256((__m256i const*)codes);
		__m256i c = loadu_si256i(codes);
		codes += 32;

		__m256i mask = _mm256_set1_epi8(0xf);
		__m256i code_hi = _mm256_srli_epi16(c,4);
		code_hi = _mm256_and_si256(code_hi, mask);
		__m256i code_lo = _mm256_and_si256(c,mask);
		// __m256i code_lo = _mm256_set1_epi8(0x1);
		// Log<uint8_t>(c);
		for (int q = 0; q < Number_Query; q++) {
			// load LUTs for 2 quantizers
			// __m256i lut = _mm256_load_si256((__m256i const*)LUT);
			__m256i lut = loadu_si256i(LUT);
			LUT += 32;
			__m256i res0 = _mm256_shuffle_epi8(lut, code_hi);
			__m256i res1 = _mm256_shuffle_epi8(lut, code_lo);
     
	        // Log<uint8_t> (lut);
            // Log<uint8_t> (code_hi);
            // Log<uint8_t> (code_lo);
            // Log<uint8_t> (res0);
            // Log<uint8_t> (res1);
			// printf("\n");

			accu[q][0] += res0;
			accu[q][1] += _mm256_srli_epi16(res0,8);
			accu[q][2] += res1;
			accu[q][3] += _mm256_srli_epi16(res1,8);
		}
	}
	printf("[kernel] start load norm\n");

	__m256i temp_norm0 = loadu_si256i(norms);
	norms+=16;
	__m256i temp_norm1 = loadu_si256i(norms);

	// Log<uint16_t> (temp_norm0);
	// Log<uint16_t> (temp_norm1);

    // __m256i temp_norm0 = _mm256_set1_epi16(0x01);
    // __m256i temp_norm1 = _mm256_set1_epi16(0x01);


	for (int q = 0; q < Number_Query; q++) {
		accu[q][0] -= _mm256_slli_epi16(accu[q][1], 8 );
		__m256i dis_norm0 = rbe_combine2x2(accu[q][0], accu[q][1], temp_norm0);
		accu[q][2] -= _mm256_slli_epi16(accu[q][3], 8 );
		__m256i dis_norm1 = rbe_combine2x2(accu[q][2], accu[q][3], temp_norm1);
		_mm256_storeu_si256((__m256i*)res, dis_norm0);
		_mm256_storeu_si256((__m256i*)(res+16), dis_norm1);
		// printf("%f",dis_norm0)
		// printf("%u\n", (unsigned int)res[0]);
		// printf("%u\n", (unsigned int)res[1]);
	}
}

void IndexRbeScan::convert(int n, int bit_num, const uint8_t* x, std::vector<uint8_t>& codes) {
	int code8_num = bit_num/8;
	for (int i=0;  i< code8_num*n; i++ ) {
		uint8_t first_code = (x[i] & 240)>>4;
		uint8_t second_code = x[i] & 15;
		codes.push_back(first_code);
		codes.push_back(second_code);
		printf(" %i %u %u %u ||", i , x[i],first_code,second_code);
		// printf(" %u %u ||",first_code,second_code);
	}
	printf("\n");
	return ;
}

IndexRbeScan::IndexRbeScan(int nbits) {
	M2 = nbits/4;
	// the sub-space number
	return ;
}
int IndexRbeScan::add(int n, int bit_num, const float* x) {
	int code4_nums = bit_num/4;
	int code8_nums = bit_num/8;
	ntotal = roundup(n,32);
	std::cout << ntotal<<std::endl;
	int col_num = 32;
	int row_num = (ntotal/32) * M2 /2;
	int lack_num = ntotal - n;
	std::cout<< "total number:"<< ntotal << " lack num:" << lack_num<<std::endl;
	std::cout<< "bit_num:"<< bit_num <<" code8_nums:" << code8_nums <<std::endl;
	std::cout<< "col_num:"<< col_num <<" row_num:" << row_num <<std::endl;
	std::vector<float> temp_x;
	for (int i=0; i<n; i++) {
		for (int j=0; j<code8_nums+1; j++) {
			temp_x.push_back(x[i*(code8_nums+1)+j]);
		}
	}

	for (int i=0; i<lack_num; i++) {
		for (int j=0; j<code8_nums+1; j++ ) {
			temp_x.push_back(0);
		}
	}
	
	std::cout << "finish the temp data prepare"<< std::endl;
	// std::cout << temp_x[33] << std::endl;
	for (int i=0; i<row_num ; i = i+1) {
 		for (int j=0; j < col_num/2; j++ ) {
			uint8_t first_code = (uint8_t) temp_x[ (j*2)*(code8_nums + 1) + i ];
			uint8_t second_code = (uint8_t) temp_x[ (j*2+1)*(code8_nums+1) + i];
			uint8_t the_code = (first_code & 240) |  ((second_code>>4) & 15);
			// if ((i==0) ){
				// printf("%u,%u,%u,%u,%u",first_code,second_code ,first_code&240 , ((second_code>>4) & 15), the_code);
				// printf("\n"); 
				// 28: 00011100  28,76,16,4,20  00010000 01000000>>4
				// printf("%u,", (first_code&240)>>4);
			// }
			codes.push_back(the_code);
		}
		printf(" || ");
		for (int j=0; j < col_num/2; j++ ) {
			uint8_t first_code = (uint8_t) temp_x[ (j*2)*(code8_nums + 1) + i ];
			uint8_t second_code = (uint8_t) temp_x[ (j*2+1)*(code8_nums+1) + i];
			uint8_t the_code = ((first_code & 15) << 4 ) |  (second_code & 15);
			// if ((i==0)){
				// printf("%u,%u,%u,%u,%u",first_code,second_code ,((first_code & 15) << 4 ) ,  (second_code & 15), the_code);
				// printf("\n");
				// printf("%u,",(first_code & 15));
			// }
			codes.push_back(the_code);
		}
		// printf("\n");
	}

	for (int i=0; i<ntotal;i++) {
		norms.push_back( (uint16_t) (temp_x[ (i+1) * (code8_nums+1)-1]*512) );
		printf(" %u,", norms[i]);
	}
	std::cout << "finish the temp data prepare"<< std::endl;
	return 0;
}
void IndexRbeScan::search(int n, const float *x, int bit_num, int k) {
	const int QBS = 1;
	int code4_nums = bit_num / 4;
	int code8_nums = bit_num / 8;
	int nsq = bit_num / 4;
	// 256/4 =32
	constexpr int Q1 = QBS & 15;
	std::cout<< "num of querys:" <<n<< "ntotal:"<<ntotal<<std::endl;

	std::cout<< "start to compute the distance" <<std::endl;

    std::vector<uint16_t> res;
    res.resize(ntotal);
    std::vector<int> index(res.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }

	const uint16_t* Norm = &norms[0];
	const uint8_t* Code = &codes[0];

	std::vector<uint8_t> temp_x;
	for (int i=0; i< n; i++) {
		for (int j=0; j<code8_nums; j++) {
			temp_x.push_back( (uint8_t)x[i*(code8_nums+1)+j] );
		}
	}
	for (int i=0;i <n; i=i+QBS) {
        uint16_t* Res = res.data();

		std::vector<uint8_t> lut_index_code;
		convert(QBS, bit_num, temp_x.data()+i*(code8_nums+1), lut_index_code);
		std::cout<< "lut_index_code shape" <<lut_index_code.size()<<std::endl;
		for (int i=0; i < QBS * code4_nums ; i++) {
			for (int j =0; j<16; j++) {
				lookuptable.push_back(centroids[lut_index_code[i]*16 + j]);
				// printf(" %u, ",centroids[lut_index_code[i]*16 + j]);
			}
            // printf("\n");
		}
		std::cout << "lookuptable shape: " << lookuptable.size() << std::endl;
		for (int64_t j0 = 0; j0 < ntotal; j0 += 32) {
			const uint8_t* LUT = &lookuptable[0];
			rbe_kernel_scan<Q1>(nsq, Code, Norm, LUT, Res);
			LUT += Q1 * nsq * 16;
			Code += 32 * nsq / 2;
			Norm += 32;
			Res += 32;
		}


        for (int i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }

        sort(index.begin(), index.end(),
            [&](const int& a, const int& b) {
            return (res[a] > res[b]);
        }
        );
        for (int i = 0 ; i != index.size() ; i++) {
            std::cout << "index:"<<index[i] << " distance:" << res[index[i]]<<" , ";
        }
        res.clear();

		lut_index_code.clear();
		lookuptable.clear();
		std::vector<uint8_t> ().swap(lut_index_code);
		std::vector<uint8_t> ().swap(lookuptable);

        std::cout<<std::endl;
		printf("\n");

	}



	return ;
}
int main() {
	int nq=1;
	//default nq = 31642
	int k=40;
	// nb of results per query in the GT
	int nb=nq*k;
	// default nb = 629331
	int bit_nums=256;
	int depth = 2;
	printf("[rbe] prepare train/databse/query set\n");
	std::string query_codebook =
	            "../data/python/q_rbe_uint8.txt";
	std::string db_codebook =
	            "../data/python/db_rbe_uint8.txt";
	std::vector<float> xq = load_codebook_file(query_codebook,nq);
	std::vector<float> xb =  load_codebook_file(db_codebook,nb);
	// for(int i=0; i<xq.size();i++){
	// std::cout<<xq[i]<<std::endl;
	// }
	IndexRbeScan idx_rbe(256);
	idx_rbe.add(nb,bit_nums,xb.data());
	idx_rbe.search(nq,xq.data(),bit_nums,k);
	return 0;
}