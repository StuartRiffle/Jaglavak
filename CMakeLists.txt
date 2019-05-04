cmake_minimum_required( VERSION 3.8 )

project( Jaglavak LANGUAGES CXX CUDA )
include_directories( Source )

set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --disable-warnings -cudart static" )
set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

add_library( X64    STATIC Source/Player/X64.cpp )
add_library( SSE4   STATIC Source/Player/SSE4.cpp )
add_library( AVX2   STATIC Source/Player/AVX2.cpp )
add_library( AVX512 STATIC Source/Player/AVX512.cpp )
add_library( CUDA   STATIC Source/Player/CudaKernel.cu )

target_compile_options( SSE4   PRIVATE -msse4.1 )
target_compile_options( AVX2   PRIVATE -mavx2 )
target_compile_options( AVX512 PRIVATE -mavx512f -mavx512dq )

add_executable( Jaglavak Source/Jaglavak.cpp )
target_compile_options( Jaglavak PUBLIC -fopenmp -std=c++11 )
target_link_libraries( Jaglavak X64 SSE4 AVX2 AVX512 CUDA )
