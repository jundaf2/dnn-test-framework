//
// Created by jundafeng on 3/7/22.
//
#ifndef TEST_CUDA_GLOBAL_POINTER_KERNELS_CUH
#define TEST_CUDA_GLOBAL_POINTER_KERNELS_CUH

#define CATCH_CONFIG_CPP11_TO_STRING
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN

#include "common/catch.hpp"
#include "common/fmt.hpp"
#include "common/utils.hpp"


#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

static bool verify(const std::vector<uint32_t>& expected, const std::vector<uint32_t>& actual) {
   // INFO("Verifying the output");
   SECTION("the expected and actual sizes must match") {
      REQUIRE(expected.size() == actual.size());
   }

   SECTION("the results must match") {
      for (int ii = 0; ii < expected.size(); ii++) {
         INFO("the results did not match at index " << ii);
         REQUIRE(expected[ii] == actual[ii]);
      }
   }
   return true;
}

__global__ void vec_add_kernel(uint32_t* in1, uint32_t* in2, uint32_t* out, int len){
   int idx = blockDim.x*blockIdx.x+threadIdx.x;
   if(idx<len){
      out[idx] = in1[idx]+in2[idx];
   }
}

#endif //TEST_CUDA_GLOBAL_POINTER_KERNELS_CUH
