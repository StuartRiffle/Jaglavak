// Job.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_JOB_H__
#define CORVID_JOB_H__

struct PlayoutJobInfo
{
    Position            mPosition;
    PlayoutOptions      mOptions;
    int                 mNumGames;
    MoveList            mPathFromRoot;
};

struct PlayoutJobResult
{
    MoveList    mPathFromRoot;
    ScoreCard   mScores;
};



#if !CORVID_CUDA_DEVICE
typedef std::shared_ptr< PlayoutJobInfo >       PlayoutJobInfoRef;
typedef std::shared_ptr< PlayoutJobResult >     PlayoutJobResultRef;

typedef ThreadSafeQueue< PlayoutJobInfoRef >    PlayoutJobQueue;
typedef ThreadSafeQueue< PlayoutJobResultRef >  PlayoutResultQueue;
#endif

#endif
