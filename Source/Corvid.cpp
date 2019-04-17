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
#include "TreeNode.h"
#include "CpuWorker.h"
#include "GpuWorker.h"
#include "TreeSearcher.h"
#include "UciEngine.h"

EvalTerm EvaluateFenPosition( const char* fen )
{
    Position pos;
    StringToPosition( fen, pos );

    MoveMap moveMap;
    pos.CalcMoveMap( &moveMap );

    EvalWeightSet weights;

    float gamePhase = Evaluation::CalcGamePhase( pos );
    Evaluation::GenerateWeights( &weights, gamePhase );

    u64 score = Evaluation::EvaluatePosition( pos, moveMap, weights );
    EvalTerm result = (EvalTerm) score;

    printf( "(%d) %s\n", result, fen );
    return result;
}

int main( int argc, char** argv )
{
    printf( "CORVID CHESS %d.%d.%d\n", CORVID_VER_MAJOR, CORVID_VER_MINOR, CORVID_VER_PATCH );
    printf( "Stuart Riffle\n\n" );

    if( !PlatDetectPopcnt() )
    {
        printf( "POPCNT support is required\n" );
        return -1;
    }

    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    UciEngine engine;

    engine.ProcessCommand( "uci" );
    //engine.ProcessCommand( "position fen 1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - -" ); // bm d6d1
//    engine.ProcessCommand( "position fen 3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - -" ); // bm d5
    //engine.ProcessCommand( "position fen r3qb1k/1b4p1/p2pr2p/3n4/Pnp1N1N1/6RP/1B3PP1/1B1QR1K1 w - -" );// bm Nxh6; id "Nolot.1";
    engine.ProcessCommand( "position fen r2qk2r/ppp1b1pp/2n1p3/3pP1n1/3P2b1/2PB1NN1/PP4PP/R1BQK2R w KQkq -" );// bm Nxg5; id "Nolot.3";

    //engine.ProcessCommand( "position fen r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq -" ); // bm h5f7 mate in one
    //engine.ProcessCommand( "position fen r1b2k1r/ppp1bppp/8/1B1Q4/5q2/2P5/PPP2PPP/R3R1K1 w - - 1 0" ); // bm d5d8 mate in two
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


