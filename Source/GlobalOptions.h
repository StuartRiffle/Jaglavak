// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

union GlobalOptions
{
    struct
    {
        // These are exposed as UCI settings

        int     _EnableSimd;
        int     _EnableCuda;
        int     _EnableMulticore;
        int     _CpuAffinityMask;
        int     _GpuAffinityMask;

        int     _DrawsWorthHalf;
        int     _NumInitialPlayouts;
        int     _NumAsyncPlayouts;
        int     _MaxPlayoutMoves;
        int     _MaxBranchExpansion;

        int     _BatchSize;
        int     _MaxPendingBatches;
        int     _MaxTreeNodes;

        int     _CudaHeapMegs;
        int     _CudaBatchesPerLaunch;
        int     _NumSimdWorkers;
        int     _TimeSafetyBuffer;
        int     _SearchSleepTime;
        int     _UciUpdateDelay;

        // These are not

        float   _ExplorationFactor;
        int     _DetectedSimdLevel;
        int     _ForceSimdLevel;
    };

    int _Option[1];
};

