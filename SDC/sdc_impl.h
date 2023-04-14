
#include <vector>

class IndexRbeScan{

    public:
    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default

    // packed version of the codes
    int ntotal;
    int M2;


    std::vector<uint8_t> lookuptable;

    std::vector<uint8_t> codes;
    
    std::vector<uint16_t> norms;


    IndexRbeScan(int nbits);

    void convert(int n, int bit_nums, const uint8_t* x, std::vector<uint8_t>& codes);

    int add(int n,int bit_nums, const uint8_t* x);

    void search(int n, const uint8_t *x, int bit_num, int k);


};