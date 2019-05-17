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
    mOptions.mVirtualLoss       = 0;
    mOptions.mVirtualLossDecay  = 1.0f;//0.999f;

    mSearcher = unique_ptr<  TreeSearch >( new TreeSearch( &mOptions ) );
    mSearcher->Init();
}

const UciOptionInfo* UciEngine::GetOptionInfo()
{
    #define OPTION_INDEX( _FIELD ) (offsetof( GlobalOptions, _FIELD ) / sizeof( int ))

    static UciOptionInfo sOptions[] = 
    {
        OPTION_INDEX( mEnableSimd ),        "EnableSimd",            0, 1, 1,
        OPTION_INDEX( mEnableCuda ),        "EnableCuda",            0, 1, 1,
        OPTION_INDEX( mEnableMulticore ),   "EnableMulticore",       0, 1, 1,
        OPTION_INDEX( mMaxTreeNodes ),      "MaxTreeNodes",         0, 1000000000, 1000000,
        OPTION_INDEX( mNumInitialPlayouts ),   "NumInitialPlayouts",   0, 64, 0,
        OPTION_INDEX( mNumAsyncPlayouts ),     "NumAsyncPlayouts",     0, 10000, 100,
        OPTION_INDEX( mBatchSize ),         "BatchSize",            1, 128, 4096,
        OPTION_INDEX( mCudaQueueDepth ),    "CudaQueueDepth",       0, 8192, 128,
        OPTION_INDEX( mPlayoutMaxMoves ),   "PlayoutMaxMoves",      0, 1000, 300,
        OPTION_INDEX( mMaxPendingJobs ),    "MaxPendingJobs",       0, 1000000, 128,
        OPTION_INDEX( mNumCpuWorkers ),    "NumCpuWorkers",      1, 10, 1,
        OPTION_INDEX( mDrawsWorthHalf ),    "DrawsWorthHalf",      1, 10, 0,

        -1
    };

    #undef OPTION_INDEX
    return sOptions;
}

void UciEngine::SetDefaultOptions()
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
        mOptions.mOption[info->mIndex] = info->mDefault;
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
            if( (option->mMin == 0) && (option->mMax == 1) )
                printf( "option name %s type check default %s\n", option->mName, option->mDefault? "true" : "false" );
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

        if( conf.mNodesLimit )
            printf( "info string WARNING: limiting by node count is not supported\n" );

        mSearcher->SetUciSearchConfig( conf );
        mSearcher->StartSearching();
    }
    else if( t.Consume( "stop" ) )
    {
        mSearcher->StopSearching();
    }
    else if( t.Consume( "ponderhit" ) )
    {
        printf( "info string WARNING: ponderhit not supported\n" );
    }
    else if( t.Consume( "quit" ) )
    {
        return true;
    }
    else
    {
        printf( "info string ERROR: invalid command\n" );
    }

    return false;
}

