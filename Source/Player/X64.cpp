// X64.cpp - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Chess.h"
#include "GamePlayer.h"
#include "PlayoutJob.h"

extern _CDECL ScoreCard PlayGamesX64( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< u64 > player( &job.mOptions, job.mRandomSeed );

    result->mPathFromRoot = job->mPathFromRoot;
    result->mScores += player.PlayGames( job.mPosition, count );
}
