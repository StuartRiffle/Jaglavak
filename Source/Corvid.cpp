// Corvid.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#include <stdio.h>
#include <ctype.h>
#include <time.h>
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

using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::list;


#include "Serialization.h"
#include "Tokenizer.h"
#include "Perft.h"
#include "Options.h"
#include "Threads.h"
#include "PlayoutJob.h"
#include "LocalWorker.h"
#include "TreeSearcher.h"
#include "UciEngine.h"
 
int main( int argc, char** argv )
{
    printf( "Corvid %d.%d.%d\n\n", CORVID_VER_MAJOR, CORVID_VER_MINOR, CORVID_VER_PATCH );

    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    UciEngine engine;

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


