// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

union GlobalOptions
{
    struct
    {
        // These are exposed as UCI settings

        int     mEnableSimd;
        int     mEnableCuda;
        int     mEnableMulticore;
        int     mMaxTreeNodes;
        int     mMaxPendingJobs;
        int     mCudaQueueDepth;
        int     mNumSimdWorkers;

        int     mNumInitialPlayouts;
        int     mNumAsyncPlayouts;
        int     mExplorationFactor;
        int     mPlayoutMaxMoves;
        int     mDrawsWorthHalf;
        int     mBatchSize;

        // These are not

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};

