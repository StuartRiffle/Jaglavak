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

        int     mDrawsWorthHalf;
        int     mMaxInitialPlayouts;
        int     mMaxAsyncPlayouts;
        int     mMaxPlayoutMoves;
        int     mMinPendingBatches;
        int     mMaxPendingBatches;
        int     mMinBatchSize;
        int     mMaxBatchSize;
        int     mMaxTreeNodes;
        int     mNumWarmupLevels;

        int     mCudaQueueDepth;
        int     mNumCpuWorkers;
        int     mTimeSafetyBuffer;
        int     mSearchSleepTime;
        int     mUciUpdateDelay;

        // These are not

        float   mExplorationFactor;
        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};

