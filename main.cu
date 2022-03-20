#include "funcs.h"
#include "kernels.cuh"
#include "global.h"

#define PRINT(...) LOG(info, std::string(fmt::format(__VA_ARGS__)))

int devQuery(){
   int deviceCount;

   cudaGetDeviceCount(&deviceCount);

   timer_start("Getting GPU Data."); //@@ start a timer

   for (int dev = 0; dev < deviceCount; dev++) {
      cudaDeviceProp deviceProp;

      cudaGetDeviceProperties(&deviceProp, dev);

      if (dev == 0) {
         if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            PRINT("No CUDA GPU has been detected");
            return -1;
         } else if (deviceCount == 1) {
            //@@ WbLog is a provided logging API (similar to Log4J).
            //@@ The logging function wbLog takes a level which is either
            //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or trace and a
            //@@ message to be printed.
            PRINT("There is 1 device supporting CUDA");
         } else {
            PRINT("There are {} devices supporting CUDA", deviceCount);
         }
      }

      PRINT("Device {} name {}", dev, deviceProp.name);
      PRINT("\tComputational Capabilities: {}.{}", deviceProp.major, deviceProp.minor);
      PRINT("\tMaximum global memory size: {}", deviceProp.totalGlobalMem);
      PRINT("\tMaximum constant memory size: {}", deviceProp.totalConstMem);
      PRINT("\tMaximum shared memory size per block: {}", deviceProp.sharedMemPerBlock);
      PRINT("\tMaximum block dimensions: {}x{}x{}", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
      PRINT("\tMaximum grid dimensions: {}x{}x{}", deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
      PRINT("\tWarp size: {}", deviceProp.warpSize);
   }

   timer_stop(); //@@ stop the timer
}

extern uint32_t* device_vec_in1;
extern uint32_t* device_vec_in2;
extern uint32_t* device_vec_out;

int eval(int inputLength = 10000) {
   timer_start("vec addition");
   auto hostInput1 = generate_input(inputLength);
   auto hostInput2 = generate_input(inputLength);
   auto gpuInput1 = hostInput1;
   auto gpuInput2 = hostInput2;

   const size_t byteCount = inputLength * sizeof(uint32_t);


   cudaMalloc((void**) &device_vec_in1, byteCount);
   cudaMalloc((void**) &device_vec_in2, byteCount);
   cudaMalloc((void**) &device_vec_out, byteCount);

   for(int i=0;i<10;i++){
      cudaMemcpy(device_vec_in1, gpuInput1.data(), byteCount,cudaMemcpyHostToDevice);
      cudaMemcpy(device_vec_in2, gpuInput2.data(), byteCount,cudaMemcpyHostToDevice);
      cudaMemset(device_vec_out, 0, byteCount);

      dim3 threadsPerBlock(512);
      dim3 numBlock((inputLength-1)/512+1);
      vec_add_kernel<<<threadsPerBlock,numBlock>>>(device_vec_in1,device_vec_in2,device_vec_out, inputLength);
      cudaDeviceSynchronize();

      std::vector<uint32_t> cpuOutput(inputLength);
      std::vector<uint32_t> gpuOutput(inputLength);
      cudaMemcpy(gpuOutput.data(), device_vec_out, byteCount, cudaMemcpyDeviceToHost);

      vec_add(hostInput1.data(), hostInput2.data(), cpuOutput.data(), inputLength);
      verify(gpuOutput, cpuOutput);

      // next iteration
      hostInput1 = cpuOutput;
      hostInput2 = generate_input(inputLength);
      gpuInput1 = gpuOutput;
      gpuInput2 = hostInput2;
   }

   cudaFree(device_vec_in1);
   cudaFree(device_vec_in2);
   cudaFree(device_vec_out);
   timer_stop();
   return 0;
}

TEST_CASE("Vec_Add", "[vec_add]") {
   SECTION("[inputSize:16191]") {
      eval(161910);
   }
}
