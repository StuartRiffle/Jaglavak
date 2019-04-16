// Options.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_OPTIONS_H__
#define CORVID_OPTIONS_H__

union GlobalOptions
{
    struct
    {
        // These are exposed as UCI settings

        int     mAllowSimd;
        int     mAllowCuda;
        int     mAllowParallel;
        int     mMaxCpuCores;
        int     mMaxTreeNodes;
        int     mNumInitialPlays;
        int     mNumAsyncPlays;
        int     mExplorationFactor;
        int     mCudaStreams;
        int     mCudaQueueDepth;
        int     mCudaJobBatch;
        int     mPlayoutPeekMoves;
        int     mPlayoutErrorRate;
        int     mPlayoutMaxMoves;
        int     mMaxPendingJobs;
        int     mNumCpuWorkers;


        // These are not 

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
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


#endif // CORVID_OPTIONS_H__

