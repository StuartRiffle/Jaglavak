// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutParams
{
    u64     mRandomSeed;
    int     mMaxMovesPerGame;
    int     mEnableMulticore;
};

struct PlayoutRequest
{
    Position        mPosition;
    PlayoutParams   mParams;
};

struct PlayoutResult
{
    ScoreCard       mScores;
};

