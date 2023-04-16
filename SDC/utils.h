#include <algorithm>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include<fstream>  


#include <immintrin.h>
#include <iostream>
#include <iomanip>    

template<class T> 
inline void Log(const __m256i & value)
{
    const size_t n = sizeof(__m256i) / sizeof(T);
    T buffer[n];
    _mm256_storeu_si256((__m256i*)buffer, value);
    for (int i = 0; i < n; i++)
        printf("%u,",buffer[i]);
        // std::cout << buffer[i] << " ";
    printf("\n");
}


void split_string_label(std::string str, std::vector<int>& vec_data){
	std::vector<std::string> lineItems;
	boost::split(lineItems,str,boost::is_any_of(" "));
    for (int i = 0; i < lineItems.size(); i++) {
        if(i==0)
            continue;
        int feat_val = int (boost::lexical_cast<float>(lineItems[i]));
        vec_data.push_back(feat_val);
    }
}

void split_string(std::string str, std::vector<float>& vec_data){
	std::vector<std::string> lineItems;
	boost::split(lineItems,str,boost::is_any_of(","));
	// feat_len = lineItems.size()-1;
    // printf("%d %d\n", lineItems.size(),lineItems.size()/128);
    // if( lineItems.size()!=128){
    //     printf("there is unmatched line\n");
    // }
    for (int i = 0; i < lineItems.size(); i++) {
        // uint8_t feat_val =
        // boost::numeric_cast<uint8_t>(boost::lexical_cast<int>(lineItems[i]));
        // printf("%s\n", lineItems[i]);
        float feat_val = boost::lexical_cast<float>(lineItems[i]);
        // feat_val = feat_val / 2;
        vec_data.push_back(feat_val);
    }
}

std::vector<int> load_codebook_file_label(std::string codebook_file,int num_code){

	std::ifstream frd(codebook_file.c_str());
	std::vector<int> codebook_vals;
	std::string line;
    int num_line = 0;
    while (getline(frd, line)) {
        if (line == "") {
            continue;
        }
        if (num_line > num_code) {
            break;
        } else {
            num_line = num_line + 1;
        }
        split_string_label(line, codebook_vals);
    }
    return codebook_vals;
}

std::vector<float> load_codebook_file(std::string codebook_file,int num_code){

	std::ifstream frd(codebook_file.c_str());
	std::vector<float> codebook_vals;
	std::string line;
    int num_line = 0;
    while (getline(frd, line)) {
        if (line == "") {
            continue;
        }
        if (num_line>num_code){
            break;
        }
        else{
            num_line = num_line + 1;
        }
        split_string(line, codebook_vals);
    }
    return codebook_vals;
}
