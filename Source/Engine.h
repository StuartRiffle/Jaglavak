// Engine.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_ENGINE_H__
#define CORVID_ENGINE_H__



struct SearchConfig
{
    int                 mWhiteTimeLeft;   
    int                 mBlackTimeLeft;   
    int                 mWhiteTimeInc;    
    int                 mBlackTimeInc;    
    int                 mTimeControlMoves;
    int                 mMateSearchDepth; 
    int                 mDepthLimit;       
    int                 mNodesLimit;       
    int                 mTimeLimit; 
    MoveList            mLimitMoves;

    SearchConfig()      { this->Clear(); }
    void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};


struct EngineOptionInfo
{
    const char* mName;
    int         mMin;
    int         mMax;
    int         mDefault;
};

struct Engine
{
    bool mDebugMode;

    Engine() : mDebugMode( false ) {}

    const EngineOptionInfo* GetOptionInfo()
    {
        static EngineOptionInfo sOptions[] = 
        {
            "OwnBook", 0, 1, 0,
            NULL
        };

        return sOptions;
    }

    void SetOption( const char* name, int value )
    {
    }

    void SetDebug( bool debugMode )
    {
        mDebugMode = debugMode;
    }

    void Init()
    {
        this->Reset();
    }

    void Reset()
    {
       // mTree = new TreeSearcher( &mSearchOptions );
    }

    void SetPosition( const Position& pos )
    {
    }

    void Go( SearchConfig* conf )
    {
    }

    void Stop()
    {
    }

    void MakeMove( const char* movetext )
    {
    }
};

#endif // CORVID_ENGINE_H__

