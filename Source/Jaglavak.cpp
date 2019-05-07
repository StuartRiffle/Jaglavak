// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "UciEngine.h"

int main( int argc, char** argv )
{
    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    printf( "JAGLAVAK CHESS ENGINE %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH );
    printf( "Stuart Riffle\n\n" );

    PTR< UciEngine > engine( new UciEngine() );

    while( !feof( stdin ) )
    {
        char buf[8192];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool timeToExit = engine->ProcessCommand( cmd );
        if( timeToExit )
            break;
    }

    return 0;
}


