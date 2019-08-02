// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include <stdint.h>
#include <assert.h>
#include <omp.h>
#include <cuda_runtime_api.h>

#ifndef NDEBUG
#define DEBUG (1)
#endif

#ifndef ENABLE_POPCNT
#define ENABLE_POPCNT (0)
#endif

#if defined( __CUDA_ARCH__ )

    // We are running __device__ code

    #define ON_CUDA_DEVICE      (1)
    #define ALIGN( _N )         __align__( _N )
    #define RESTRICT            __restrict
    #define INLINE              __forceinline__    
    #define PDECL               __device__

    typedef uint64_t atomic64_t;

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

    typedef std::atomic< uint64_t > atomic64_t;

#elif defined( _MSC_VER )

    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN    
    #include <windows.h>
    #include <process.h>
    #include <intrin.h>
    #include <limits.h>

    #pragma warning( disable: 4996 )    // CRT security warnings
    #pragma warning( disable: 4293 )    // Shift count negative or too big (due to unused branch in templated function)
    #pragma warning( disable: 4752 )    // Found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
    #pragma warning( disable: 4554 )    // Check operator precedence for possible error; use parentheses to clarify precedence (false warning caused by nvcc compile)
    #pragma warning( disable: 4200 )    // Nonstandard extension used: zero-sized array in struct/union
    #pragma inline_recursion( on )
    #pragma inline_depth( 255 )
    
    #define TOOLCHAIN_MSVC      (1)
    #define ALIGN( _N )         __declspec( align( _N ) )
    #define ALIGN_SIMD          ALIGN( SIMD_ALIGNMENT )
    #define RESTRICT            __restrict
    #define DEBUGBREAK          __debugbreak
    #define INLINE              __forceinline
    #define PRId64              "I64d"
    #define PDECL         

    #include <atomic>
    typedef std::atomic< uint64_t > atomic64_t;
    
    extern "C" void * __cdecl memset(void *, int, size_t);
    #pragma intrinsic( memset )        

#else
    #error
#endif


typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;

#define DEBUG_LOG   printf

#define SIMD_WIDEST     (8)
#define SIMD_ALIGNMENT  (SIMD_WIDEST * sizeof( uint64_t ))
#define ALIGN_SIMD      ALIGN( SIMD_ALIGNMENT )

#define MIN( _A, _B ) (((_A) < (_B))? (_A) : (_B))
#define MAX( _A, _B ) (((_A) > (_B))? (_A) : (_B))
#define NUM_ELEMENTS( _ARR ) (sizeof( _ARR ) / sizeof( _ARR[0] ))


INLINE PDECL u64 PlatByteSwap64( const u64& val )             
{ 
#if ON_CUDA_DEVICE
    u32 hi = __byte_perm( (u32) val, 0, 0x0123 );
    u32 lo = __byte_perm( (u32) (val >> 32), 0, 0x0123 );
    return( ((u64) hi << 32ULL) | lo );
#elif TOOLCHAIN_GCC
    return( __builtin_bswap64( val ) );     
#elif TOOLCHAIN_MSVC
    return( _byteswap_uint64( val ) ); 
#endif
}

INLINE PDECL u64 PlatLowestBitIndex64( const u64& val )
{
#if ON_CUDA_DEVICE
    return( __ffsll( val ) - 1 );
#elif TOOLCHAIN_GCC
    return( __builtin_ffsll( val ) - 1 ); 
#elif TOOLCHAIN_MSVC
    unsigned long result;
    _BitScanForward64( &result, val );
    return( result );
#endif
}

INLINE PDECL void PlatStoreAtomic( atomic64_t* dest, u64 val )
{
#if ON_CUDA_DEVICE
    atomicExch( (unsigned long long*) dest, val );
#else
    dest->store( val );
#endif
}

INLINE PDECL u64 PlatAddAtomic( atomic64_t* dest, u64 val )
{
#if ON_CUDA_DEVICE
    atomicAdd( (unsigned long long*) dest, val );
#else
    dest->fetch_add( val );
#endif
    return (u64) *dest;
}

INLINE PDECL u64 PlatCountBits64( u64 n )
{
#if ON_CUDA_DEVICE
    return( __popcll( n ) );
#elif ENABLE_POPCNT
    #if TOOLCHAIN_GCC
        return( __builtin_popcountll( n ) );
    #elif TOOLCHAIN_MSVC
        return( __popcnt64( n ) );
    #endif
#else
    const u64 mask01 = 0x0101010101010101ULL;
    const u64 mask0F = 0x0F0F0F0F0F0F0F0FULL;
    const u64 mask33 = 0x3333333333333333ULL;
    const u64 mask55 = 0x5555555555555555ULL;
    n =  n - ((n >> 1) & mask55);
    n = (n & mask33) + ((n >> 2) & mask33);
    n = (n + (n >> 4)) & mask0F;
    n = (n * mask01) >> 56;
    return( n );
#endif
}

INLINE PDECL int PlatDetectCpuCores()
{
#if ON_CUDA_DEVICE
    return 1;
#elif TOOLCHAIN_GCC
    return(sysconf( _SC_NPROCESSORS_ONLN ));
#elif TOOLCHAIN_MSVC
    SYSTEM_INFO si ={ 0 };
    GetSystemInfo( &si );
    return(si.dwNumberOfProcessors);
#endif
}

void PlatSetCoreAffinity( u64 mask, bool entireProcess = false );
void PlatLimitCores( int count, bool entireProcess = false );
void PlatSetThreadName( const char* name );
void PlatSleep( int ms );
