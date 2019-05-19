cmake_minimum_required( VERSION 3.8 )
project( Jaglavak LANGUAGES CXX CUDA )

set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --disable-warnings -cudart static" )
set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )


if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )

	set( FLAGS_COMMON  )
	set( FLAGS_AVX2    )
	set( FLAGS_AVX512  )

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )

	set( FLAGS_COMMON -msse4.1 -fopenmp )
	set( FLAGS_AVX2   -mavx2 )
	set( FLAGS_AVX512 -mavx512f -mavx512dq -mavx512bw )

endif()


add_library( X64    STATIC Source/Player/X64.cpp )
add_library( SSE4   STATIC Source/Player/SSE4.cpp )
add_library( AVX2   STATIC Source/Player/AVX2.cpp )
add_library( AVX512 STATIC Source/Player/AVX512.cpp )
add_library( CUDA   STATIC Source/Player/CudaKernel.cu )

target_compile_options( X64      PRIVATE ${FLAGS_COMMON} )
target_compile_options( SSE4     PRIVATE ${FLAGS_COMMON} )
target_compile_options( AVX2     PRIVATE ${FLAGS_COMMON} ${FLAGS_AVX2} )
target_compile_options( AVX512   PRIVATE ${FLAGS_COMMON} ${FLAGS_AVX512} )

include_directories( 
    Source 
    Source/Misc
    Source/Player
    Source/Worker
)

add_executable( Jaglavak 
    Source/Jaglavak.cpp
    Source/TreeNode.cpp
    Source/TreeSearch.cpp
    Source/UciEngine.cpp
    Source/Worker/CudaWorker.cpp
)

target_compile_options( Jaglavak PUBLIC ${FLAGS_COMMON} -std=c++11 )
target_link_options(    Jaglavak PUBLIC -fopenmp )
target_link_libraries(  Jaglavak PUBLIC
    X64 
    SSE4 
    AVX2 
    AVX512 
    CUDA
)
