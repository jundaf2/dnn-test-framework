cmake_minimum_required(VERSION 3.20)

project(test_cuda_global_pointer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(DEFAULT_BUILD_TYPE "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)
add_executable(test_cuda_global_pointer
        main.cu
        funcs.h
        kernels.cuh
        common/fmt.hpp
        common/catch.hpp
        common/clara.hpp
        common/utils.hpp
        global.h)

target_link_libraries(test_cuda_global_pointer ${CUDA_LIBRARIES})
include_directories(test_cuda_global_pointer ${CUDA_INCLUDE_DIRS})
set_target_properties(test_cuda_global_pointer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
