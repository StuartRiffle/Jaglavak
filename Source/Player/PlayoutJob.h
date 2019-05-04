// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutOptions
{

};

struct PlayoutJob
{
    Position        mPosition;
    MoveList        mPathFromRoot;
    u64             mRandomSeed;
    int             mNumGames;
    int             mMaxMoves;
};

struct PlayoutResult
{
    ScoreCard       mScores;
    MoveList        mPathFromRoot;

    float           mCpuLatency;
    float           mGpuTime;
};

typedef ThreadSafeQueue< PlayoutJob >    PlayoutJobQueue;
typedef ThreadSafeQueue< PlayoutResult > PlayoutResultQueue;

struct CudaLaunchSlot
{
    cudaStream_t    mStream;                    /// Stream this job was issued into
    cudaEvent_t     mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t     mReadyEvent;                /// GPU timer event to notify that the results have been copied back to host memory

    PlayoutJob*     mInputHost;                 /// Job input buffer, host side
    PlayoutResult*  mOutputHost;                /// Job output buffer, host side

    PlayoutJob*     mInputDev;                  /// Job input buffer, device side
    PlayoutResult*  mOutputDev;                 /// Job output buffer, device side

    u64             mTickQueued;                /// CPU tick when the job was queued for execution
};

