// UCI.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_UCI_H__
#define CORVID_UCI_H__

struct UciEngine
{
    TreeSearcher    mSearcher;
    EngineOptions   mOptions;
    bool            mDebugMode;

public:
    UciEngine() : mDebugMode( false ) 
    {
        this->SetDefaultOptions();
    }

    const EngineOptionInfo* GetOptionInfo()
    {
        static EngineOptionInfo sOptions[] = 
        {
            OPTION_ENABLE_POPCNT, "EnablePopcnt", 0, 1, 1,
            OPTION_ENABLE_SIMD,   "EnableSimd",   0, 1, 1,
            OPTION_ENABLE_CUDA,   "EnableCuda",   0, 1, 1,
            -1
        };

        return sOptions;
    }

    void SetDefaultOptions()
    {
        for( const EngineOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
            mOptions[info->mIndex] = info->mDefault;
    }

    void SetOptionByName( const char* name, int value )
    {
        for( const EngineOptionInfo* info = GetOptionInfo(); info->mIndex >= 0; info++ )
        {
            if( !stricmp( name, info->mName ) )
            {
                mOptions.mOption[info->mIndex] = value;
                break;
            }
        }
    }

    bool ProcessCommand( Engine* engine, const char* cmd )
    {
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
            mSearcher.Reset();
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
                    if( !FEN::StringToMoveSpec( movetext, move ) )
                        printf( "info string ERROR: unable to parse move" );

                    moveList.Append( move );
                }
            }

            mSearcher.SetPosition( pos, moveList );
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

            mSearcher.StartSearching( conf );
        }
        else if( t.Consume( "stop" ) )
        {
            mSearcher.StopSearching();
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
};





#endif CORVID_UCI_H__

