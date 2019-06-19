// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

union GlobalOptions
{
    struct
    {
        int     _EnableSimd;
        int     _EnableCuda;
        int     _EnableMulticore;
        int     _CpuWorkThreads;
        int     _CpuAffinityMask;
        int     _CudaAffinityMask;
        int     _CudaHeapMegs;
        int     _CudaBatchSize;
        int     _DrawsWorthHalf;
        int     _NumPlayoutsAtLeaf;
        int     _MaxPlayoutMoves;
        int     _CpuBatchSize;
        int     _MaxTreeNodes;
        int     _TimeSafetyBuffer;
        int     _UciUpdateDelay;
        int     _FlushEveryBatch;
        int     _BranchesToExpandAtLeaf;
        int     _FixedRandomSeed;
        int     _CpuSearchFibers;
        int     _ExplorationFactor;
        int     _DetectedSimdLevel;
        int     _ForceSimdLevel;
    };

    int _OptionByIndex[];

    void Initialize( const vector< string >& configFiles );
    void SetOptionByName( const char* name, int value );
};

struct OptionInfo
{
    int         _Index;
    const char* _Name;
    int         _Value;
};



