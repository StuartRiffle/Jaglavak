// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutRequest
{
    Position    mPosition;
    u64         mRandomSeed;
    int         mMaxMovesPerGame;
    int         mEnableMulticore;
};

struct PlayoutResult
{
    ScoreCard   mScores;
};

