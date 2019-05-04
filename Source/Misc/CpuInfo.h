// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct CpuInfo
{
    static bool CheckCpuFlag( int leaf, int idxWord, int idxBit )
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

    static int DetectSimdLevel()
    {
        bool avx512 = CheckCpuFlag( 7, 1, 16 ) && CheckCpuFlag( 7, 1, 17 );   
        if( avx512 )
            return( 8 );

        bool avx2 = CheckCpuFlag( 7, 1, 5 );   
        if( avx2 )
            return( 4 );

        bool sse4 = CheckCpuFlag( 1, 2, 19 );
        if( sse4 )
            return( 2 );

        return( 1 );
    }

    static int DetectCpuCores()
    {
    #if TOOLCHAIN_MSVC
        SYSTEM_INFO si = { 0 };
        GetSystemInfo( &si );
        return( si.dwNumberOfProcessors );
    #elif TOOLCHAIN_GCC
        return( sysconf( _SC_NPROCESSORS_ONLN ) );
    #endif
    }

    static INLINE u64 GetClockTick()
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

    static INLINE u64 GetClockFrequency()
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
};
