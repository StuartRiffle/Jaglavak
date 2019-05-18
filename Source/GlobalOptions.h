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
        int     mMaxCores;

        int     mDrawsWorthHalf;
        int     mNumInitialPlayouts;
        int     mNumAsyncPlayouts;
        int     mPlayoutMaxMoves;

        int     mMaxTreeNodes;
        int     mBatchSize;
        int     mMaxPendingBatches;
        int     mCudaQueueDepth;
        int     mNumCpuWorkers;

        // These are not

        float   mExplorationFactor;

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};

