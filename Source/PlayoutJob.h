// Job.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_JOB_H__
#define CORVID_JOB_H__

struct PlayoutJob
{
    Position        mPosition;
    GlobalOptions   mOptions;
    int             mNumGames;
    MoveList        mPathFromRoot;
};

struct PlayoutResult
{
    ScoreCard   mScores;
    MoveList    mPathFromRoot;
};



#if !CORVID_CUDA_DEVICE
typedef std::shared_ptr< PlayoutJob >       PlayoutJobRef;
typedef ThreadSafeQueue< PlayoutJobRef >    PlayoutJobQueue;

typedef std::shared_ptr< PlayoutResult >    PlayoutResultRef;
typedef ThreadSafeQueue< PlayoutResultRef > PlayoutResultQueue;
#endif

#endif
