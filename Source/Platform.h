// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define VERSION_MAJOR (1000)
#define VERSION_MINOR (0)
#define VERSION_PATCH (1)

#if defined( __CUDA_ARCH__ )

    // We are running __device__ code

    #include <cuda_runtime_api.h>

    #define ON_CUDA_DEVICE      (1)
    #define ALIGN( _N )         __align__( _N )
    #define RESTRICT            __restrict
    #define INLINE              __forceinline__    
    #define PDECL               __device__

#elif defined( _MSC_VER )

    #define WIN32_LEAN_AND_MEAN    
    #include <windows.h>
    #include <process.h>
    #include <intrin.h>
    #include <limits.h>

    #pragma warning( disable: 4996 )    // CRT security warnings
    #pragma warning( disable: 4293 )    // Shift count negative or too big (due to unused branch in templated function)
    #pragma warning( disable: 4752 )    // Found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
    #pragma warning( disable: 4554 )    // Check operator precedence for possible error; use parentheses to clarify precedence (false warning caused by nvcc compile)
    #pragma inline_recursion( on )
    #pragma inline_depth( 255 )
    
    #define TOOLCHAIN_MSVC      (1)
    #define ALIGN( _N )         __declspec( align( _N ) )
    #define RESTRICT            __restrict
    #define DEBUGBREAK          __debugbreak
    #define INLINE              __forceinline
    #define PRId64              "I64d"
    #define PDECL         

    extern "C" void * __cdecl memset(void *, int, size_t);
    #pragma intrinsic( memset )        

#elif defined( __GNUC__ )

    #define __STDC_FORMAT_MACROS

    #include <inttypes.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <x86intrin.h>
    #include <cpuid.h>
    #include <string.h>
    #include <unistd.h>
    #include <sched.h>

    #pragma GCC diagnostic ignored "-Wunknown-pragmas"

    #define TOOLCHAIN_GCC       (1)
    #define ALIGN( _N )         __attribute__(( aligned( _N ) ))
    #define RESTRICT            __restrict
    #define DEBUGBREAK          void
    #define INLINE              inline __attribute__(( always_inline ))
    #define PDECL         

    #define stricmp             strcasecmp
    #define strnicmp            strncasecmp

#else
    #error
#endif

    #include <stdint.h>


#if 1//!ON_CUDA_DEVICE

    #include <assert.h>
    #include <stdio.h>
    #include <ctype.h>
    #include <time.h>
    #include <math.h>
    #include <algorithm>
    #include <string>
    #include <vector>
    #include <list>
    #include <map>
    #include <functional>
    #include <memory>
    #include <thread>
    #include <mutex>
    #include <condition_variable>
    #include <atomic>

    using std::pair;
    using std::vector;
    using std::map;
    using std::list;
    using std::string;    
    using std::lock_guard;
    using std::mutex;
    using std::condition_variable;
    using std::thread;

    template< typename T > class PTR : public std::unique_ptr< T > {};
    template< typename T > class RC  : public std::shared_ptr< T > {};

#endif

#define ALIGN_SIMD  ALIGN( 64 ) // worst case (AVX-512)
#define DEBUG_LOG   printf

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;


INLINE PDECL u64 PlatByteSwap64( const u64& val )             
{ 
#if ON_CUDA_DEVICE
    u32 hi = __byte_perm( (u32) val, 0, 0x0123 );
    u32 lo = __byte_perm( (u32) (val >> 32), 0, 0x0123 );
    return( ((u64) hi << 32ULL) | lo );
#elif TOOLCHAIN_MSVC
    return( _byteswap_uint64( val ) ); 
#elif TOOLCHAIN_GCC
    return( __builtin_bswap64( val ) );     
#endif
}

INLINE PDECL u64 PlatLowestBitIndex64( const u64& val )
{
#if ON_CUDA_DEVICE
    return( __ffsll( val ) - 1 );
#elif TOOLCHAIN_MSVC
    unsigned long result;
    _BitScanForward64( &result, val );
    return( result );
#elif TOOLCHAIN_GCC
    return( __builtin_ffsll( val ) - 1 ); 
#endif
}

INLINE PDECL void PlatSleep( int ms )
{
#if TOOLCHAIN_MSVC
    Sleep( ms );
#elif TOOLCHAIN_GCC
    timespec request;
    timespec remaining;
    request.tv_sec  = (ms / 1000);
    request.tv_nsec = (ms % 1000) * 1000 * 1000;
    nanosleep( &request, &remaining );
#endif
}


