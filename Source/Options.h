// Options.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_OPTIONS_H__
#define CORVID_OPTIONS_H__

enum 
{
    OPTION_ENABLE_POPCNT,
    OPTION_ENABLE_SIMD,  
    OPTION_ENABLE_CUDA, 

    OPTION_COUNT
};

union EngineOptions
{
    struct
    {
        int mEnablePopcnt;
        int mEnableSimd;
        int mEnableCuda;

    };

    int mOption[1];
};

struct UciOptionInfo
{
    int         mIndex;
    const char* mName;
    int         mMin;
    int         mMax;
    int         mDefault;
};

struct UciSearchConfig
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

    UciSearchConfig()   { this->Clear(); }
    void Clear()        { PlatClearMemory( this, sizeof( *this ) ); }
};

struct PlayoutOptions
{
    int     mErrorRate;
    int     mMovesToPeek;
    int     mMaxPlayoutMoves;
    int     mAutoAdjudicate;
    u64     mRandomSeed;

    int     mMaxCpuLevel;
    int     mForceCpuLevel;
    bool    mUsePopcnt;
};


struct SearchOptions
{
    int mMaxTreeNodes;
    int mNumInitialPlays;
    int mNumAsyncPlays;
    int mTreeUpdateBatch;

    PlayoutOptions mPlayout;
};

                                       


#endif // CORVID_OPTIONS_H__

