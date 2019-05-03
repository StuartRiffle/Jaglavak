cmake_minimum_required( VERSION 3.8 )

project( Corvid LANGUAGES CXX CUDA )

if( ${CMAKE_CXX_COMPILER_ID} MATCHES MSVC )

	add_compile_options( /GS- /GL /W3 /Gy- /Zc:wchar_t /Zi /Gm- /Ox /Ob2 /fp:fast /fp:except- /WX- /GR- /Gd /Oy /Oi /MT )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG" )

elseif( (${CMAKE_CXX_COMPILER_ID} MATCHES GNU) OR (${CMAKE_CXX_COMPILER_ID} MATCHES Clang) )

	# -fsanitize=address,undefined -fsanitize-address-use-after-scope -fstack-check -fcf-protection=full -fstack-protector-all 
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -w -std=c++11 -Wl,--no-export-dynamic -msse4.1 -fno-omit-frame-pointer" )

endif()

set( CMAKE_CUDA_SEPARABLE_COMPILATION ON )
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --disable-warnings -cudart static -use_fast_math" )

add_library( CudaPlayer STATIC
	Source/PlayoutKernel.cu )

add_executable( Corvid 
	Source/Corvid.cpp )

target_link_libraries( 
	Corvid 
	CudaPlayer )
