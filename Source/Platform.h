// Platform.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLATFORM_H__
#define CORVID_PLATFORM_H__

#include <stdint.h>
#include <assert.h>

#define SUPPORT_CUDA (1)

#if SUPPORT_CUDA
#include <cuda_runtime_api.h>
#endif

#if SUPPORT_CUDA && defined( __CUDA_ARCH__ )

    // We are running __device__ code

    #define RUNNING_ON_CUDA_DEVICE  (1)
    #define ALIGN( _N )  __align__( _N )
    #define ALIGN_SIMD   __align__( 32 )    

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
    
    #define RUNNING_ON_CPU     (1)
    #define TOOLCHAIN_MSVC      (1)
    #define SUPPORT_SSE4         (1)
    #define SUPPORT_AVX2         (1)
    #define SUPPORT_AVX512       (0)
    #define ALIGN( _N )  __declspec( align( _N ) )
    #define ALIGN_SIMD   __declspec( align( 32 ) )

    #define RESTRICT            __restrict
    #define DEBUGBREAK          __debugbreak
    #define INLINE              __forceinline
    #define PDECL         
    #define PRId64              "I64d"

    extern "C" void * __cdecl memset(void *, int, size_t);
    #pragma intrinsic( memset )        

#elif defined( __GNUC__ )

    #define __STDC_FORMAT_MACROS

    #include <inttypes.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <emmintrin.h>
    #include <cpuid.h>
    #include <string.h>
    #include <unistd.h>
    #include <sched.h>
    #include <atomic>

    #pragma GCC diagnostic ignored "-Wunknown-pragmas"

    #define RUNNING_ON_CPU          (1)
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


#if SUPPORT_CUDA && RUNNING_ON_CPU
#define RUNNING_ON_CUDA_HOST (1)
#endif

typedef uint64_t  u64;
typedef int64_t   i64;
typedef uint32_t  u32;
typedef int32_t   i32;
typedef uint16_t  u16;
typedef int16_t   i16;
typedef uint8_t   u8;
typedef int8_t    i8;

enum
{
    CPU_SCALAR,
    CPU_SSE4,
    CPU_AVX2,
    CPU_AVX512,

    CPU_LEVELS,
    CPU_INVALID
};

INLINE PDECL int PlatGetSimdWidth( int cpuLevel )
{
    assert( cpuLevel < CPU_LEVELS );

    int width[] = { 1, 2, 2, 4, 8 };
    return( width[cpuLevel] );
}

INLINE PDECL u64 PlatByteSwap64( const u64& val )             
{ 
#if RUNNING_ON_CUDA_DEVICE
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
#if RUNNING_ON_CUDA_DEVICE
    return( __ffsll( val ) - 1 );
#elif TOOLCHAIN_MSVC
    unsigned long result;
    _BitScanForward64( &result, val );
    return( result );
#elif TOOLCHAIN_GCC
    return( __builtin_ffsll( val ) - 1 ); 
#endif
}

INLINE PDECL u64 PlatCountBits64( const u64& val )
{
#if RUNNING_ON_CUDA_DEVICE
    return( __popcll( val ) );
#elif TOOLCHAIN_MSVC
    return( __popcnt64( val ) );
#elif TOOLCHAIN_GCC
    return( __builtin_popcountll( val ) );
#endif
}

INLINE PDECL void PlatClearMemory( void* mem, size_t bytes )
{
#if RUNNING_ON_CUDA_DEVICE
    memset( mem, 0, bytes );
#elif TOOLCHAIN_MSVC
    ::memset( mem, 0, bytes );
#elif TOOLCHAIN_GCC
    __builtin_memset( mem, 0, bytes );    
#endif
}

#if !RUNNING_ON_CUDA_DEVICE

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

INLINE PDECL bool PlatDetectPopcnt()
{
#if TOOLCHAIN_GCC
    #if defined( __APPLE__ )
        return( false ); // FIXME: Apple LLVM 8.0 does not provide __builtin_cpu_supports()
    #else
        return( __builtin_cpu_supports( "popcnt" ) );
    #endif
#else
    return( PlatCheckCpuFlag( 1, 2, 23 ) );
#endif
}

INLINE PDECL int PlatDetectSimdLevel()
{
#if SUPPORT_AVX2
    bool avx2 = PlatCheckCpuFlag( 7, 1, 5 );   
    if( avx2 )
        return( CPU_AVX2 );
#endif

#if SUPPORT_SSE4
    bool sse4 = PlatCheckCpuFlag( 1, 2, 20 );
    if( sse4 )
        return( CPU_SSE4 );
#endif

    return( CPU_SCALAR );
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

INLINE PDECL void PlatSetThreadName( const char* name )
{
#if PIGEON_MSVC
    //#pragma pack( push, 8 )
    struct THREADNAME_INFO
    {
        DWORD   dwType;     
        LPCSTR  szName;     
        DWORD   dwThreadID; 
        DWORD   dwFlags;    
    };
    //#pragma pack( pop )

    THREADNAME_INFO info;

    info.dwType     = 0x1000;
    info.szName     = name;
    info.dwThreadID = GetCurrentThreadId();
    info.dwFlags    = 0;

    __try
    {
        const DWORD MS_VC_EXCEPTION = 0x406D1388;
        RaiseException( MS_VC_EXCEPTION, 0, sizeof( info ) / sizeof( ULONG_PTR ), (ULONG_PTR*) &info );
    }
    __except( EXCEPTION_EXECUTE_HANDLER ) {}
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
    sched_yield();
#endif
}

static INLINE u64 PlatGetClockTick()
{ 
#if TOOLCHAIN_MSVC
    LARGE_INTEGER tick; 
    QueryPerformanceCounter( &tick ); 
    return( tick.QuadPart ); 
#elif TOOLCHAIN_GCC
    return( (u64) clock() );
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
    return( (u64) CLOCKS_PER_SEC );
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

struct Timer
{
    u64     mStartTime;

    Timer() { this->Reset(); }
    Timer( const Timer& rhs ) : mStartTime( rhs.mStartTime ) {}

    void    Reset()         { mStartTime = Timer::GetTick(); }
    i64     GetElapsedMs()  { return( ((i64) (Timer::GetTick() - mStartTime) * 1000) / Timer::GetFrequency() ); }

    static INLINE u64 GetTick()
    { 
#if TOOLCHAIN_MSVC
        LARGE_INTEGER tick; 
        QueryPerformanceCounter( &tick ); 
        return( tick.QuadPart ); 
#elif TOOLCHAIN_GCC
        return( (u64) clock() );
#endif
    }

    static u64 GetFrequency()
    {
#if TOOLCHAIN_MSVC
        static LARGE_INTEGER freq = { 0 };
        if( !freq.QuadPart )
            QueryPerformanceFrequency( &freq );
        return( freq.QuadPart );
#elif TOOLCHAIN_GCC
        return( (u64) CLOCKS_PER_SEC );
#endif
    }
};

#endif // !RUNNING_ON_CUDA_DEVICE
#endif // CORVID_PLATFORM_H__
