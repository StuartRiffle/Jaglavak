cmake_minimum_required( VERSION 3.8 )
project( Jaglavak LANGUAGES CXX CUDA )

if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )
	set( FLAGS_SSE4             -msse4.1 )
	set( FLAGS_AVX2             -mavx2 )
	set( FLAGS_AVX512           -mavx512f -mavx512dq -mavx512bw )
	set( FLAGS_COMMON           -fopenmp )
    set( CMAKE_EXE_LINKER_FLAGS -fopenmp )
elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
	set( FLAGS_SSE4             /arch:SSE2 )
    set( FLAGS_AVX2             /arch:AVX2 )
	set( FLAGS_AVX512           /arch:AVX2 )
	set( FLAGS_COMMON )
endif()

include_directories( 
    Source 
    Source/Util
    Source/TreeSearch
)
add_executable( Jaglavak Source/**/*.cpp )
#    Source/Jaglavak.cpp
#    Source/TreeSearch/TreeNode.cpp
#    Source/TreeSearch/TreeSearch.cpp
#   Source/UciEngine.cpp
#   Source/CUDA/CudaWorker.cpp
#)
add_library( X64 STATIC Source/SIMD/X64.cpp )
target_compile_options( X64 PRIVATE ${FLAGS_COMMON} )

add_library( SSE4 STATIC Source/SIMD/SSE4.cpp )
target_compile_options( SSE4 PRIVATE ${FLAGS_COMMON} ${FLAGS_SSE4} )

add_library( AVX2 STATIC Source/SIMD/AVX2.cpp )
target_compile_options( AVX2 PRIVATE ${FLAGS_COMMON} ${FLAGS_AVX2} )

add_library( AVX512 STATIC Source/SIMD/AVX512.cpp )
target_compile_options( AVX512 PRIVATE ${FLAGS_COMMON} ${FLAGS_AVX512} )
                               `
add_library( CUDA STATIC Source/CUDA/CudaPlayer.cu )
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --disable-warnings -cudart static" )
set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )

target_compile_options( Jaglavak PUBLIC ${FLAGS_COMMON} -std=c++11 )
target_link_libraries(  Jaglavak PUBLIC X64 SSE4 AVX2 AVX512 CUDA )

