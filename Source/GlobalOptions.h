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

        int     mNumInitialPlays;
        int     mNumAsyncPlays;
        int     mExplorationFactor;
        int     mPlayoutMaxMoves;
        int     mDrawsHaveValue;
        int     mBatchSize;

        // These are not

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};

