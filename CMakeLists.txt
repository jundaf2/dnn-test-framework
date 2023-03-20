cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

set(CMAKE_CUDA_STANDARD 14)
set(ENABLE_CUDA ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(DEFAULT_BUILD_TYPE "Release")

project(nnTest LANGUAGES C CXX CUDA)
set(${PROJECT_NAME}_VERSION 0.0.0)
enable_language(C CXX CUDA)

if (NOT CMAKE_CUDA_HOST_COMPILER)
	set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR " does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

set(COMMON_FILES_DIR ${PROJECT_SOURCE_DIR}/common)
set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)


set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}" CACHE STRING "CUDA flags set" FORCE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -no-pie -lpthread")

add_compile_options(-Wno-unknown-pragmas)

### Find Required Libs ###
find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

# Find CUDA ARCH VERSION
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET( ${CUDA_ARCH_LIST})
list (GET CUDA_ARCH_LIST 0 ARCH_NUMBER)
SET(CUDA_ARCH sm_${ARCH_NUMBER})
SET(CUDA_COMPUTE compute_${ARCH_NUMBER})
message(--------------------------------------------)
message(CUDA_ARCH = ${CUDA_ARCH})
message(CUDA_COMPUTE = ${CUDA_COMPUTE})
message(--------------------------------------------)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

set(CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr --compiler-options -fPIC -g -G -Xcompiler -Wall")
set(CUDA_FLAGS "-gencode code=${CUDA_ARCH},arch=${CUDA_COMPUTE} ${CUDA_FLAGS}")
string(REGEX REPLACE "^sm_" "" ARCH_NUMBER "${CUDA_ARCH}")
if ("${CUDA_ARCH}" STREQUAL "sm_${ARCH_NUMBER}")
	set(CMAKE_CUDA_ARCHITECTURES "${ARCH_NUMBER}")
else()
	message(FATAL_ERROR "Unknown CUDA_ARCH: ${CUDA_ARCH}")
endif()
message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

set(CUDA_FLAGS "-ccbin=${CMAKE_CXX_COMPILER} ${CUDA_FLAGS}")


# Headers and sources
set(MAIN_FILE ${PROJECT_SOURCE_DIR}/nn_test.h ${PROJECT_SOURCE_DIR}/linear_layer_kernels.cu ${PROJECT_SOURCE_DIR}/linear_layer.h ${PROJECT_SOURCE_DIR}/main.cc ) #  
# Declaring the files
set(ALL_SRC ${MAIN_FILE})
foreach(FNAME IN LISTS ALL_SRC)
    message(${FNAME})
endforeach()

include(CTest)
add_executable(nntest ${ALL_SRC})
target_include_directories(nntest PUBLIC ${COMMON_FILES_DIR} SYSTEM ${CUDA_INCLUDE_DIRS}) #  ${TORCH_INCLUDE_DIRS}
target_link_directories(nntest PRIVATE ${CMAKE_MODULE_PATH})
target_link_libraries(nntest PRIVATE ${CUDA_LIBRARIES} CUDA::cublas CUDA::cublasLt ${TORCH_LIBRARIES})

set_target_properties(nntest PROPERTIES POSITION_INDEPENDENT_CODE ON)

