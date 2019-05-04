// Platform.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#pragma once

#include <stdint.h>
#include <assert.h>

#include <cuda_runtime_api.h>

#if defined( __CUDA_ARCH__ )

    // We are running __device__ code

    #define ON_CUDA_DEVICE      (1)
    #define ALIGN( _N )         __align__( _N )
    #define ALIGN_SIMD          __align__( 8 )    

    #define RESTRICT            __restrict
    #define INLINE              __forceinline__    
    #define PDECL               __device__

#elif defined( _MSC_VER )

    #define WIN32_LEAN_AND_MEAN    
    #include <windows.h>
    #include <process.h>
    #include <intrin.h>
    #include <limits.h>
    #include <atomic>

    #pragma warning( disable: 4996 )    // CRT security warnings
    #pragma warning( disable: 4293 )    // Shift count negative or too big (due to unused branch in templated function)
    #pragma warning( disable: 4752 )    // Found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
    #pragma warning( disable: 4554 )    // Check operator precedence for possible error; use parentheses to clarify precedence (false warning caused by nvcc compile)
    #pragma inline_recursion( on )
    #pragma inline_depth( 255 )
    
    #define TOOLCHAIN_MSVC      (1)
    #define ALIGN( _N )         __declspec( align( _N ) )
    #define ALIGN_SIMD          __declspec( align( 32 ) )
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
    #include <atomic>

    #pragma GCC diagnostic ignored "-Wunknown-pragmas"

    #define TOOLCHAIN_GCC       (1)
    #define ALIGN( _N )  __attribute__(( aligned( _N ) ))
    #define ALIGN_SIMD   __attribute__(( aligned( 32 ) ))    

    #define RESTRICT            __restrict
    #define DEBUGBREAK          void
    #define INLINE              inline __attribute__(( always_inline ))
    #define PDECL         

    #define stricmp             strcasecmp
    #define strnicmp            strncasecmp

#else
    #error
#endif

#define DEBUG_LOG printf

typedef uint64_t  u64;
typedef int64_t   i64;
typedef uint32_t  u32;
typedef int32_t   i32;
typedef uint16_t  u16;
typedef int16_t   i16;
typedef uint8_t   u8;
typedef int8_t    i8;


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

INLINE PDECL void PlatClearMemory( void* mem, size_t bytes )
{
#if ON_CUDA_DEVICE
    memset( mem, 0, bytes );
#elif TOOLCHAIN_MSVC
    ::memset( mem, 0, bytes );
#elif TOOLCHAIN_GCC
    __builtin_memset( mem, 0, bytes );    
#endif
}

#if !ON_CUDA_DEVICE

INLINE PDECL bool PlatCheckCpuFlag( int leaf, int idxWord, int idxBit )
{
#if TOOLCHAIN_MSVC
    int info[4] = { 0 };
    __cpuid( info, leaf );
#elif TOOLCHAIN_GCC
    unsigned int info[4] = { 0 };
    if( !__get_cpuid( leaf, info + 0, info + 1, info + 2, info + 3 ) )
        return( false );
#endif

    return( (info[idxWord] & (1 << idxBit)) != 0 );
}

INLINE PDECL int PlatDetectSimdLevel()
{
    bool avx512 = PlatCheckCpuFlag( 7, 1, 16 ) && PlatCheckCpuFlag( 7, 1, 17 );   
    if( avx512 )
        return( 8 );

    bool avx2 = PlatCheckCpuFlag( 7, 1, 5 );   
    if( avx2 )
        return( 4 );

    bool sse4 = PlatCheckCpuFlag( 1, 2, 19 );
    if( sse4 )
        return( 2 );

    return( 1 );
}

INLINE PDECL int PlatDetectCpuCores()
{
#if TOOLCHAIN_MSVC
    SYSTEM_INFO si = { 0 };
    GetSystemInfo( &si );
    return( si.dwNumberOfProcessors );
#elif TOOLCHAIN_GCC
    return( sysconf( _SC_NPROCESSORS_ONLN ) );
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

INLINE PDECL void PlatYield()
{
#if TOOLCHAIN_MSVC
    Sleep( 1 );
#elif TOOLCHAIN_GCC
    //sched_yield();
    PlatSleep( 1 );
#endif
}

static INLINE u64 PlatGetClockTick()
{ 
#if TOOLCHAIN_MSVC
    LARGE_INTEGER tick; 
    QueryPerformanceCounter( &tick ); 
    return( tick.QuadPart ); 
#elif TOOLCHAIN_GCC
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    return( (ts.tv_sec * 1000000000) + ts.tv_nsec );    
#endif
}

static u64 PlatGetClockFrequency()
{
#if TOOLCHAIN_MSVC
    static LARGE_INTEGER freq = { 0 };
    if( !freq.QuadPart )
        QueryPerformanceFrequency( &freq );
    return( freq.QuadPart );
#elif TOOLCHAIN_GCC
    return( 1000000000 );
#endif
}

static void PlatBoostThreadPriority()
{
#if TOOLCHAIN_MSVC
    SetThreadPriority( GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL );
#elif TOOLCHAIN_GCC
    // TODO
#endif
}

#endif // !ON_CUDA_DEVICE
