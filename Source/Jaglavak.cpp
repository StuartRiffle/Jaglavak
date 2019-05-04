// Jaglavak.cpp - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <list>
#include <map>
#include <functional>
#include <memory>
#include <thread>

#include "Misc/Serialization.h"
#include "Misc/Tokenizer.h"
#include "Misc/Perft.h"
#include "Misc/Threads.h"

#include "Options.h"
#include "PlayoutJob.h"
#include "PlayoutCpu.h"
#include "TreeNode.h"
#include "LocalWorker.h"
#include "CudaWorker.h"
#include "TreeSearcher.h"
#include "UciEngine.h"

int main( int argc, char** argv )
{
    printf( "JAGLAVAK CHESS %d.%d.%d\n", JAGLAVAK_VER_MAJOR, JAGLAVAK_VER_MINOR, JAGLAVAK_VER_PATCH );
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
        char buf[UCI_COMMAND_BUFFER];

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


