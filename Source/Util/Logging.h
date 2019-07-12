#pragma once

static mutex sLogMutex;
static Timer sLogTimer;

INLINE ConsolePrint( const char* str )
{
    unique_lock< mutex > lock( sLogMutex );
    printf( str );
}

INLINE void Log( const char* fmt, ... )
{
    const int BUFSIZE = 1024;
    char str[BUFSIZE];

    va_list args;
    va_start( args, fmt );
    vsprintf( str, fmt, args );
    va_end( args );

    LogInternal( str );
}


INLINE void LogInternal( const char* str )
{
    int t   = sLogTimer.GetElapsedMicros();
    int us  = t % 1000000;
    int sec = t / 1000000;

    char 

    string usstr = format( "%-6d", us );
    

    string uci = format( "info string | %3d.%6d (+.%s) | " + (sec?)


    // info string |   3.442945 (+0...3929) |  [CPU] Intel Xeon
    // info string |   3.443295 (+0....929) |  [CPU] Intel Xeon

}
