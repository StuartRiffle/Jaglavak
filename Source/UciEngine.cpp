// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

#include "CpuInfo.h"
#include "FEN.h"
#include "Tokenizer.h"
#include "UciEngine.h"
#include "Version.h"

UciEngine::UciEngine() : mDebugMode( false ) 
{
    this->SetDefaultOptions();

    mOptions.mDetectedSimdLevel = CpuInfo::DetectSimdLevel();
    mOptions.mForceSimdLevel    = 0;
    mOptions.mExplorationFactor = 1.41f;

    mSearcher = unique_ptr<  TreeSearch >( new TreeSearch( &mOptions ) );
    mSearcher->Init();
}

const UciOptionInfo* UciEngine::GetOptionInfo()
{
    #define OPTION_INDEX( _FIELD ) (offsetof( GlobalOptions, m##_FIELD ) / sizeof( int )), #_FIELD

    static UciOptionInfo sOptions[] = 
    {
        OPTION_INDEX( EnableSimd ),             CHECKBOX,   1,          
        OPTION_INDEX( NumSimdWorkers ),         0,          1,          
        OPTION_INDEX( EnableCuda ),             CHECKBOX,   1,          
        OPTION_INDEX( EnableMulticore ),        CHECKBOX,   1,          
        OPTION_INDEX( CpuAffinityMask ),        CHECKBOX,   0,          
        OPTION_INDEX( GpuAffinityMask ),        CHECKBOX,   1,          
        OPTION_INDEX( DrawsWorthHalf ),         CHECKBOX,   1,          
        OPTION_INDEX( NumInitialPlayouts ),     0,          0,          
        OPTION_INDEX( MaxPlayoutMoves ),        0,          200,          
        OPTION_INDEX( NumAsyncPlayouts ),       0,          1,         
        OPTION_INDEX( MaxPendingBatches ),      0,          128,        
        OPTION_INDEX( BatchSize ),              0,          128,       
        OPTION_INDEX( MaxTreeNodes ),           0,          1000000,    
        OPTION_INDEX( CudaHeapMegs ),           0,          64,        
        OPTION_INDEX( CudaBatchesPerLaunch ),   0,          8,        
        OPTION_INDEX( TimeSafetyBuffer ),       0,          100,          
        OPTION_INDEX( SearchSleepTime ),        0,          100,          
        OPTION_INDEX( UciUpdateDelay ),         0,          500,          
        -1
    };

    #undef OPTION_INDEX
    return sOptions;
}

void UciEngine::SetDefaultOptions()
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
        mOptions.mOption[info->mIndex] = info->mValue;
}

void UciEngine::SetOptionByName( const char* name, int value )
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
    {
        if( !stricmp( name, info->mName ) )
        {
            mOptions.mOption[info->mIndex] = value;
            break;
        }
    }
}

bool UciEngine::ProcessCommand( const char* cmd )
{
    Tokenizer t( cmd );

    if( t.Consume( "uci" ) )
    {                                                                                        
        printf( "id name Jaglavak %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH );
        printf( "id author Stuart Riffle\n" );

        const UciOptionInfo* option = this->GetOptionInfo();
        while( option->mIndex >= 0 )
        {
            if( option->mIsCheckbox )
                printf( "option type check name   %-20s default  %d\n", option->mName, option->mValue );
            else
                printf( "option type spin  name   %-20s default  %d\n", option->mName, option->mValue );

            option++;
        }

        printf( "uciok\n\n" );
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
            mDebugMode = true;
        else 
            mDebugMode = false;
    }
    else if( t.Consume( "isready" ) )
    {
        printf( "readyok\n" );
    }
    else if( t.Consume( "ucinewgame" ) )
    {
        mSearcher->Reset();
    }
    else if( t.Consume( "position" ) )
    {
        Position pos;
        MoveList moveList;

        pos.Reset();
        t.Consume( "startpos" );

        if( t.Consume( "fen" ) )
            if( !t.ConsumePosition( pos ) )
                printf( "info string ERROR: unable to parse FEN\n" );

        if( t.Consume( "moves" ) )
        {
            for( const char* movetext = t.ConsumeNext(); movetext; movetext = t.ConsumeNext() )
            {
                MoveSpec move;
                if( !StringToMoveSpec( movetext, move ) )
                    printf( "info string ERROR: unable to parse move" );

                moveList.Append( move );
            }
        }

        mSearcher->SetPosition( pos, &moveList );
    }
    else if( t.Consume( "go" ) )
    {
        UciSearchConfig conf;

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
                    StringToMoveSpec( movetext, spec );

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

        mSearcher->SetUciSearchConfig( conf );
        mSearcher->StartSearching();
    }
    else if( t.Consume( "stop" ) )
    {
        mSearcher->StopSearching();
    }
    else if( t.Consume( "quit" ) )
    {
        return true;
    }
    else
    {
        printf( "info string ERROR\n" );
    }

    return false;
}

