// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "UciEngine.h"

const int VER_MAJOR = 1000;
const int VER_MINOR = 0;
const int VER_PATCH = 1;

int main( int argc, char** argv )
{
    printf( "JAGLAVAK CHESS ENGINE %d.%d.%d\n", VER_MAJOR, VER_MINOR, VER_PATCH );
    printf( "Stuart Riffle\n\n" );

    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    std::unique_ptr< UciEngine > engine( new UciEngine() );

    // FIXME: testing
    engine->ProcessCommand( "uci" );
    engine->ProcessCommand( "position fen r2qk2r/ppp1b1pp/2n1p3/3pP1n1/3P2b1/2PB1NN1/PP4PP/R1BQK2R w KQkq -" );// bm Nxg5; id "Nolot.3";
    engine->ProcessCommand( "go" );

    while( !feof( stdin ) )
    {
        char buf[8192];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool timeToExit = engine->ProcessCommand( cmd );
        if( timeToExit )
            break;

        fflush( stdout );
    }

    return 0;
}


