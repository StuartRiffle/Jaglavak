// X64.cpp - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "PlayoutJob.h"
#include "GamePlayer.h"

extern _CDECL ScoreCard PlayGamesX64( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< u64 > player( &job.mOptions, job.mRandomSeed );

    result->mPathFromRoot = job->mPathFromRoot;
    result->mScores += player.PlayGames( job.mPosition, count );
}
