// Corvid.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

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

#include "Serialization.h"
#include "Tokenizer.h"
#include "Perft.h"
#include "Options.h"
#include "Threads.h"
#include "PlayoutJob.h"
#include "PlayoutCpu.h"
#include "LocalWorker.h"
#include "TreeNode.h"
#include "TreeSearcher.h"
#include "UciEngine.h"
 
int main( int argc, char** argv )
{
    printf( "CORVID CHESS %d.%d.%d\n", CORVID_VER_MAJOR, CORVID_VER_MINOR, CORVID_VER_PATCH );
    printf( "Stuart Riffle\n\n" );

    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    UciEngine engine;

    engine.ProcessCommand( "uci" );
    engine.ProcessCommand( "position fen 1rbq1rk1/p1b1nppp/1p2p3/8/1B1pN3/P2B4/1P3PPP/2RQ1R1K w - -" );
    engine.ProcessCommand( "go" );

    while( !feof( stdin ) )
    {
        char buf[UCI_COMMAND_BUFFER];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        bool timeToExit = engine.ProcessCommand( cmd );
        if( timeToExit )
            break;

        fflush( stdout );
    }

    return 0;
}


