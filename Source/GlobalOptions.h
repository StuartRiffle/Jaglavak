// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

union GlobalOptions
{
    struct
    {
        // These are exposed as UCI settings

        int     mAllowSimd;
        int     mAllowCuda;
        int     mAllowMulticore;
        int     mMaxCpuCores;
        int     mMaxTreeNodes;
        int     mMaxPendingJobs;
        int     mCudaStreams;
        int     mCudaQueueDepth;
        int     mCudaJobBatch;
        int     mNumLocalWorkers;

        int     mNumInitialPlays;
        int     mNumAsyncPlays;
        int     mExplorationFactor;
        int     mPlayoutMaxMoves;

        // These are not

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};


