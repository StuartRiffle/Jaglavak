// PlayoutJob.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_JOB_H__
#define CORVID_PLAYOUT_JOB_H__

struct PlayoutJob
{
    GlobalOptions   mOptions;
    Position        mPosition;
    u64             mRandomSeed;
    int             mNumGames;
    MoveList        mPathFromRoot;
};

struct PlayoutResult
{
    ScoreCard       mScores;
    MoveList        mPathFromRoot;
};

#if SUPPORT_CUDA
struct CudaLaunchSlot
{
    PlayoutJob      mInfo;
    PlayoutResult   mResult;
    cudaStream_t    mStream;                    /// Stream this job was issued into
    cudaEvent_t     mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t     mEndEvent;                  /// GPU timer event to notify that the results have been copied back to host memory

    PlayoutJob*     mInputHost;                 /// Job input buffer, host side
    PlayoutResult*  mOutputHost;                /// Job output buffer, host side

    PlayoutJob*     mInputDev;                  /// Job input buffer, device side
    PlayoutResult*  mOutputDev;                 /// Job output buffer, device side

    u64             mTickQueued;                /// CPU tick when the job was queued for execution
    u64             mTickReturned;              /// CPU tick when the completed job was found
    float           mCpuLatency;                /// CPU time elapsed (in ms) between those two ticks, represents job processing latency
    float           mGpuTime;                   /// GPU time spent executing kernel (in ms)
};
#endif


#endif
