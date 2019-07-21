// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Jaglavak.h"

void PlatSleep( int ms )
{
#if TOOLCHAIN_GCC
    timespec request;
    timespec remaining;
    request.tv_sec  = (ms / 1000);
    request.tv_nsec = (ms % 1000) * 1000 * 1000;
    nanosleep( &request, &remaining );
#elif TOOLCHAIN_MSVC
    Sleep( ms );
#endif
}

void PlatSetCoreAffinity( u64 mask, bool entireProcess )
{
    // TODO: support more than 64 cores   
#if TOOLCHAIN_GCC
    const cpu_set_t cs;
    CPU_ZERO( &cs );
    for( int i = 0; i < 64; i++ )
        if( mask & (1ULL << i) )
            CPU_SET( i, &cs );
    if( entireProcess )
        sched_setaffinity( gettid(), sizeof( cs ), &cs );
    else
        pthread_setaffinity_np( pthread_self(), sizeof( cs ), &cs );
#elif TOOLCHAIN_MSVC
    if( entireProcess )
        SetProcessAffinityMask( GetCurrentProcess(), mask );
    else
        SetThreadAffinityMask( GetCurrentThread(), mask );
#endif
}

void PlatLimitCores( int count, bool entireProcess )
{
    int cores = PlatDetectCpuCores();

    if( count < 0 )
        count += cores;

    u64 maskAll = (1ULL << cores) - 1;
    u64 maskSkip = (1ULL << (cores - count)) - 1;
    u64 mask = maskAll - maskSkip;

    PlatSetCoreAffinity( mask, entireProcess );
}

void PlatAdjustThreadPriority( int delta )
{
#if TOOLCHAIN_GCC
    // TODO
#elif TOOLCHAIN_MSVC
    SetThreadPriority( GetCurrentThread(), delta );
#endif
}

void PlatSetThreadName( const char* name )
{
#if TOOLCHAIN_GCC
    // TODO
#elif TOOLCHAIN_MSVC
    size_t len = strlen( name );
    LPCWSTR lpcwName = std::wstring( name, name + len ).c_str();
    //SetThreadDescription( GetCurrentThread(), lpcwName );
#endif
}
