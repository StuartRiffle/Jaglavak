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
        int     mNumCpuWorkers;

        int     mNumInitialPlayouts;
        int     mNumAsyncPlayouts;
        int     mPlayoutMaxMoves;
        int     mDrawsWorthHalf;
        int     mBatchSize;

        // These are not

        float   mExplorationFactor;
        float   mVirtualLoss;
        float   mVirtualLossDecay;

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};

