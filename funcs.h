//
// Created by jundafeng on 3/7/22.
//
#ifndef TEST_CUDA_GLOBAL_POINTER_FUNCS_H
#define TEST_CUDA_GLOBAL_POINTER_FUNCS_H
#include <cuda.h>
#include <algorithm>
#include <random>

static std::vector<uint32_t> generate_input(int len) {
   static std::random_device rd;  // Will be used to obtain a seed for the random number engine
   static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
   static std::uniform_int_distribution<> dis(1, 6);

   std::vector<uint32_t> res(len);
   std::generate(res.begin(), res.end(), [&]() {
      uint32_t r = 0;
      do {
         r = static_cast<uint32_t>(dis(gen));
      } while (r <= 0);
      return r;
   });

   return res;
}

__host__ void vec_add(uint32_t* in1, uint32_t* in2, uint32_t* out, int len){
   for(int idx=0;idx<len;idx++){
      out[idx] = in1[idx]+in2[idx];
   }
}

#endif //TEST_CUDA_GLOBAL_POINTER_FUNCS_H
