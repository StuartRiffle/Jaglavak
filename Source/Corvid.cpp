// Corvid.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

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

#include <stdio.h>
#include <ctype.h>
#include <time.h>

#include "FEN.h"
#include "Tokens.h"
#include "Perft.h"
#include "Engine.h"

using std::unique_ptr;
using std::vector;
using std::list;

int main( int argc, char** argv )
{
    setvbuf( stdin,  NULL, _IONBF, 0 );
    setvbuf( stdout, NULL, _IONBF, 0 );

    unique_ptr< Engine > engine( new Engine() );

    while( !feof( stdin ) )
    {
        char buf[8192];

        const char* cmd = fgets( buf, sizeof( buf ), stdin );
        if( cmd == NULL )
            continue;

        Tokenizer t( cmd );

        if( t.Consume( "uci" ) )
        {                                                                                        
            printf( "id name Corvid %d.%d.%d\n", CORVID_VER_MAJOR, CORVID_VER_MINOR, CORVID_VER_PATCH );
            printf( "id author Stuart Riffle\n" );

            const EngineOptionInfo* option = engine->GetOptionInfo();
            while( option->mName )
            {
                if( (option->mMin == 0) && (option->mMax == 1) )
                    printf( "option name %s type check default %d\n", option->mName, option->mDefault? "true" : "false" );
                else
                    printf( "option name %s type spin min %d max %d default %d\n", option->mName, option->mMin, option->mMax, option->mDefault );

                option++;
            }

            printf( "uciok\n" );
        }
        else if( t.Consume( "setoption" ) )
        {
            if( t.Consume( "name" ) )
            {
                const char* optionName = t.ConsumeNext();

                if( t.Consume( "value" ) )
                    engine->SetOption( optionName, t.ConsumeInt() );
            }
        }
        else if( t.Consume( "debug" ) )
        {
            if( t.Consume( "on" ) )       
                engine->SetDebug( true );
            else if( t.Consume( "off" ) ) 
                engine->SetDebug( false );
        }
        else if( t.Consume( "isready" ) )
        {
            engine->Init();
            printf( "readyok\n" );
        }
        else if( t.Consume( "ucinewgame" ) )
        {
            engine->Reset();
        }
        else if( t.Consume( "position" ) )
        {
            if( t.Consume( "startpos" ) )
                engine->Reset();

            if( t.Consume( "fen" ) )
            {
                Position pos;

                if( t.ConsumePosition( pos ) )
                    engine->SetPosition( pos );
                else
                    printf( "info string ERROR: unable to parse FEN\n" );
            }

            if( t.Consume( "moves" ) )
            {
                for( const char* movetext = t.ConsumeNext(); movetext; movetext = t.ConsumeNext() )
                    engine->MakeMove( movetext );
            }
        }
        else if( t.Consume( "go" ) )
        {
            SearchConfig conf;

            for( ;; )
            {
                if(      t.Consume( "wtime" ) )          conf.mWhiteTimeLeft       = t.ConsumeInt();
                else if( t.Consume( "btime" ) )          conf.mBlackTimeLeft       = t.ConsumeInt();
                else if( t.Consume( "winc" ) )           conf.mWhiteTimeInc        = t.ConsumeInt();
                else if( t.Consume( "binc" ) )           conf.mBlackTimeInc        = t.ConsumeInt();
                else if( t.Consume( "movestogo" ) )      conf.mTimeControlMoves    = t.ConsumeInt();
                else if( t.Consume( "mate" ) )           conf.mMateSearchDepth     = t.ConsumeInt();
                else if( t.Consume( "depth" ) )          conf.mDepthLimit          = t.ConsumeInt();
                else if( t.Consume( "nodes" ) )          conf.mNodesLimit          = t.ConsumeInt();
                else if( t.Consume( "movetime" ) )       conf.mTimeLimit           = t.ConsumeInt();
                else if( t.Consume( "infinite" ) )       conf.mTimeLimit           = 0;
                else if( t.Consume( "searchmoves" ) )
                {
                    for( const char* movetext = t.ConsumeNext(); movetext; movetext = t.ConsumeNext() )
                    {
                        MoveSpec spec;
                        FEN::StringToMoveSpec( movetext, spec );

                        conf.mLimitMoves.Append( spec );
                    }
                }
                else if( t.Consume( "ponder" ) )
                {
                    printf( "info string WARNING: pondering is not supported\n" );
                }
                else
                    break;
            }

            if( conf.mMateSearchDepth )
                printf( "info string WARNING: mate search is not supported\n" );

            if( conf.mNodesLimit )
                printf( "info string WARNING: limiting by node count is not supported\n" );

            engine->Go( &conf );
        }
        else if( t.Consume( "ponderhit" ) )
        {
            printf( "info string WARNING: ponderhit not supported\n" );
        }
        else if( t.Consume( "stop" ) )
        {
            engine->Stop();
        }
        else if( t.Consume( "quit" ) )
        {
            engine->Stop();
            break;
        }
        else
        {
            printf( "info string ERROR: invalid command\n" );
        }

        fflush( stdout );
    }

    return 0;
}


