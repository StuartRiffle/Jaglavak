// Options.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#pragma once

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
        int     mNumLocalWorkers;

        // These are not 

        int     mDetectedSimdLevel;
        int     mForceSimdLevel;
    };

    int mOption[1];
};


