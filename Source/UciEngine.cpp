// UciEngine.cpp - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

UciEngine() : mDebugMode( false ) 
{
    this->SetDefaultOptions();

    mOptions.mDetectedSimdLevel  = PlatDetectSimdLevel();
    mOptions.mForceSimdLevel     = 0;

    mSearcher = std::unique_ptr< TreeSearcher >( new TreeSearcher( &mOptions ) );
    mSearcher->Init();
}

const UciOptionInfo* GetOptionInfo()
{
    #define OPTION_INDEX( _FIELD ) (offsetof( GlobalOptions, _FIELD ) / sizeof( int ))

    static UciOptionInfo sOptions[] = 
    {
        OPTION_INDEX( mAllowSimd ),         "AllowSimd",            0, 0,    0,
        OPTION_INDEX( mAllowCuda ),         "AllowCuda",            0, 0,    0,
        OPTION_INDEX( mAllowParallel ),     "AllowParallel",        0, 0,    0,
        OPTION_INDEX( mMaxCpuCores ),       "MaxCpuCores",          0, 1024, 0,
        OPTION_INDEX( mMaxTreeNodes ),      "MaxTreeNodes",         0, 1000000000, 1000000,
        OPTION_INDEX( mNumInitialPlays ),   "NumInitialPlayouts",   0, 64,   1,
        OPTION_INDEX( mNumAsyncPlays ),     "NumAsyncPlayouts",     0, 10000, 0,//1024,
        OPTION_INDEX( mExplorationFactor ), "ExplorationFactor",    0, 1000, 141,
        OPTION_INDEX( mCudaStreams ),       "CudaStreams",          0, 16,   16,            
        OPTION_INDEX( mCudaQueueDepth ),    "CudaQueueDepth",       0, 8192, 128,
        OPTION_INDEX( mPlayoutPeekMoves ),  "PlayoutPeekMoves",     0, 1000, 0,
        OPTION_INDEX( mPlayoutErrorRate ),  "PlayoutErrorRate",     0, 100,  100,
        OPTION_INDEX( mPlayoutMaxMoves ),   "PlayoutMaxMoves",      0, 1000, 200,
        OPTION_INDEX( mMaxPendingJobs ),    "MaxPendingJobs",       0, 1000000, 128,
        OPTION_INDEX( mNumLocalWorkers ),     "NumLocalWorkers",        1, 10, 1,

        -1
    };

    #undef OPTION_INDEX
    return sOptions;
}

void SetDefaultOptions()
{
    for( const UciOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
        mOptions.mOption[info->mIndex] = info->mDefault;
}

void SetOptionByName( const char* name, int value )
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

bool ProcessCommand( const char* cmd )
{
    Tokenizer t( cmd );

    if( t.Consume( "uci" ) )
    {                                                                                        
        printf( "id name Jaglavak %d.%d.%d\n", JAGLAVAK_VER_MAJOR, JAGLAVAK_VER_MINOR, JAGLAVAK_VER_PATCH );
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

        mSearcher->StartSearching( conf );
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
