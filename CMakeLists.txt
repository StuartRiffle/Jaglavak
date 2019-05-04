cmake_minimum_required( VERSION 3.8 )

project( Jaglavak LANGUAGES CXX CUDA )
include_directories( Source )

add_library( CUDA STATIC Source/Player/CudaKernel.cu )
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --disable-warnings -cudart static -use_fast_math" )
set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

add_library( SSE4 STATIC Source/Player/SSE4.cpp )
target_compile_options( SSE4 PRIVATE -msse4.1 )

add_library( AVX2 STATIC Source/Player/AVX2.cpp )
target_compile_options( AVX2 PRIVATE -mavx2 )

add_library( AVX512 STATIC Source/Player/AVX512.cpp )
target_compile_options( AVX512 PRIVATE -mavx512f -mavx512bw )

add_executable( Jaglavak Source/Jaglavak.cpp )
target_compile_options( Jaglavak PUBLIC -fopenmp -std=c++11 )
target_link_libraries( Jaglavak CUDA SSE4 AVX2 AVX512 )
