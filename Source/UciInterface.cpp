// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "UciInterface.h"

#include "Util/FEN.h"
#include "Util/Tokenizer.h"

UciInterface::UciInterface( GlobalSettings* settings ) : _Settings( settings ), _DebugMode( false ) 
{
    _TreeSearch = unique_ptr< TreeSearch >( new TreeSearch( _Settings ) );
    _TreeSearch->Init();
}


bool UciInterface::ProcessCommand( const char* cmd )
{
    if( _DebugMode )
        printf(">>> %s\n", cmd);

    Tokenizer t( cmd );

    if( t.Consume( "uci" ) )
    {       
        cout << "id name Jaglavak " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << endl;
        cout << "id author Stuart Riffle" << endl << endl;

        _Settings->PrintListForUci();
        
        _TreeSearch->Reset();
        cout << "uciok" << endl;
    }
    else if( t.Consume( "setoption" ) )
    {
        if( t.Consume( "name" ) )
        {
            const char* optionName = t.ConsumeNext();
            if( t.Consume( "value" ) )
                _Settings->Set( optionName, t.ConsumeInt() );
        }
    }
    else if( t.Consume( "debug" ) )
    {
        if( t.Consume( "on" ) )       
            _DebugMode = true;
        else 
            _DebugMode = false;
    }
    else if( t.Consume( "isready" ) )
    {
        printf( "readyok\n" );
    }
    else if( t.Consume( "ucinewgame" ) )
    {
        _TreeSearch->Reset();
    }
    else if( t.Consume( "position" ) )
    {
        Position pos;
        MoveList moveList;

        pos.Reset();
        t.Consume( "startpos" );

        if( t.Consume( "fen" ) )
            if( !t.ConsumePosition( pos ) )
                cout << "info string ERROR: unable to parse FEN" << endl;

        if( t.Consume( "moves" ) )
        {
            for( const char* movetext = t.ConsumeNext(); movetext; movetext = t.ConsumeNext() )
            {
                MoveSpec move;
                if( !StringToMoveSpec( movetext, move ) )
                    cout << "info string ERROR: unable to parse move " << movetext << endl;

                moveList.Append( move );
            }
        }

        _TreeSearch->SetPosition( pos, &moveList );
    }
    else if( t.Consume( "go" ) )
    {
        UciSearchConfig conf;

        for( ;; )
        {
            if(      t.Consume( "wtime" ) )          conf._WhiteTimeLeft       = t.ConsumeInt();
            else if( t.Consume( "btime" ) )          conf._BlackTimeLeft       = t.ConsumeInt();
            else if( t.Consume( "winc" ) )           conf._WhiteTimeInc        = t.ConsumeInt();
            else if( t.Consume( "binc" ) )           conf._BlackTimeInc        = t.ConsumeInt();
            else if( t.Consume( "movestogo" ) )      conf._TimeControlMoves    = t.ConsumeInt();
            else if( t.Consume( "mate" ) )           conf._MateSearchDepth     = t.ConsumeInt();
            else if( t.Consume( "depth" ) )          conf._DepthLimit          = t.ConsumeInt();
            else if( t.Consume( "nodes" ) )          conf._NodesLimit          = t.ConsumeInt();
            else if( t.Consume( "movetime" ) )       conf._TimeLimit           = t.ConsumeInt();
            else if( t.Consume( "infinite" ) )       conf._TimeLimit           = 0;
            else if( t.Consume( "searchmoves" ) )
            {
                for( const char* movetext = t.ConsumeNext(); movetext; movetext = t.ConsumeNext() )
                {
                    MoveSpec spec;
                    StringToMoveSpec( movetext, spec );

                    conf._LimitMoves.Append( spec );
                }
            }
            else if( t.Consume( "ponder" ) )
            {
                cout << "info string WARNING: pondering is not supported" << endl;
            }
            else
                break;
        }

        if( conf._MateSearchDepth )
            cout << "info string WARNING: mate search is not supported" << endl;

        _TreeSearch->SetUciSearchConfig( conf );
        _TreeSearch->StartSearching();
    }
    else if( t.Consume( "stop" ) )
    {
        _TreeSearch->StopSearching();
    }
    else if( t.Consume( "quit" ) )
    {
        exit( 0 );
        //return false;
    }
    else
    {
        printf( "info string ERROR\n" );
    }

    return true;
}

