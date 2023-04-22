
#include <vector>

class IndexRbeScan{

    public:
    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default
    int ntotal;
    int ntotal32;
    int M2; // the sub-space number


    std::vector<uint8_t> lookuptable;

    std::vector<uint8_t> codes;
    
    std::vector<uint16_t> norms;


    IndexRbeScan(int nbits);

    void convert(int n, int bit_nums, const uint8_t* x, std::vector<uint8_t>& codes);

    int add(int n,int bit_nums, const float* x);

    std::vector<int> search(int n, const float *x, int bit_num, int k);


};