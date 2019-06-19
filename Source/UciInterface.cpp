// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

#include "CpuInfo.h"
#include "FEN.h"
#include "Tokenizer.h"
#include "UciInterface.h"
#include "Version.h"

UciInterface::UciInterfamce() : _DebugMode( false ) 
{
    this->SetDefaultOptions();

    _Options._DetectedSimdLevel = CpuInfo::DetectSimdLevel();
    _Options._ForceSimdLevel    = 0;
    _Options._ExplorationFactor = 1.41f;

    _Searcher = unique_ptr<  TreeSearch >( new TreeSearch( &_Options ) );
    _Searcher->Init();
}

const UciOptionInfo* UciInterface::GetOptionInfo()
{
    #define OPTION_INDEX( _FIELD ) (offsetof( GlobalOptions, _##_FIELD ) / sizeof( int )), #_FIELD

    static UciOptionInfo sOptions[] = 
    {
        OPTION_INDEX( EnableMulticore ),        0,          
        OPTION_INDEX( EnableSimd ),             0,          
        OPTION_INDEX( NumSimdWorkers ),         1,          

        OPTION_INDEX( EnableCuda ),             0,          
        OPTION_INDEX( CudaHeapMegs ),           64,        
        OPTION_INDEX( CudaBatchesPerLaunch ),   8,        
        OPTION_INDEX( GpuAffinityMask ),        1,          

        OPTION_INDEX( DrawsWorthHalf ),         1,          
        OPTION_INDEX( NumInitialPlayouts ),     8,          
        OPTION_INDEX( NumAsyncPlayouts ),       0,         
        OPTION_INDEX( MaxBranchExpansion ),     0,

        OPTION_INDEX( MaxPlayoutMoves ),        200,          
        OPTION_INDEX( MaxPendingBatches ),      32,        
        OPTION_INDEX( BatchSize ),              1024,       

        OPTION_INDEX( MaxTreeNodes ),           1000,    
        OPTION_INDEX( TimeSafetyBuffer ),       100,          
        OPTION_INDEX( SearchSleepTime ),        100,          
        OPTION_INDEX( UciUpdateDelay ),         500,          
        -1
    };

    #undef OPTION_INDEX
    return sOptions;
}


void UciInterface::SetDefaultOptions()
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->_Index >= 0; info++ )
        _Options._Option[info->_Index] = info->_Value;
}

void UciInterface::SetOptionByName( const char* name, int value )
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->_Index >= 0; info++ )
    {
        if( !stricmp( name, info->_Name ) )
        {
            _Options._Option[info->_Index] = value;
            break;
        }
    }
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

        const UciOptionInfo* option = this->GetOptionInfo();
        while( option->_Index >= 0 )
        {
            if( option->_IsCheckbox )
                cout << "option type check name " << option->_Name << " default " << option->_Value << endl;
            else
                printf( "option type spin  name   %-20s default  %d\n", option->_Name, option->_Value );

            option++;
        }
        
        _Searcher->Reset();
        cout << "uciok" << endl;
    }
    else if( t.Consume( "setoption" ) )
    {
        if( t.Consume( "name" ) )
        {
            const char* optionName = t.ConsumeNext();
            if( t.Consume( "value" ) )
                this->SetOptionByName( optionName, t.ConsumeInt() );
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
        _Searcher->Reset();
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

        _Searcher->SetPosition( pos, &moveList );
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

        _Searcher->SetUciSearchConfig( conf );
        _Searcher->StartSearching();
    }
    else if( t.Consume( "stop" ) )
    {
        _Searcher->StopSearching();
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

