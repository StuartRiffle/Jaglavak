// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct CpuInfo
{
    static void Cpuid( int leaf, unsigned int* dest )
    {
    #if TOOLCHAIN_MSVC
        __cpuid( (int*) dest leaf );
    #elif TOOLCHAIN_GCC
        __get_cpuid( leaf, dest + 0, dest + 1, dest + 2, dest + 3 );
    #endif
    }

    static bool CheckCpuFlag( int leaf, int idxWord, int idxBit )
    {
        unsigned int info[4];
        Cpuid( leaf, info );

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

    static std::string GetCpuName()
    {
        std::string result;

        union
        {
            unsigned int info[12];
            char desc[48];
        };

        Cpuid( 0x80000000, info );
        if( info[0] >= 0x80000004 )
        {
            Cpuid( 0x80000002, info );
            Cpuid( 0x80000003, info + 4 );
            Cpuid( 0x80000004, info + 8 );

            result = std::string( desc, desc + sizeof( desc ) );
        }

        return result;
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
