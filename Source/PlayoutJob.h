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



#endif
