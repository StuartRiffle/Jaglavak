// AVX2.cpp - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "AVX2.h"
#include "PlayoutJob.h"
#include "GamePlayer.h"

extern _CDECL void PlayGamesAVX2( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< simd4_avx2 > player( &job.mOptions, job.mRandomSeed );

    result->mPathFromRoot = job->mPathFromRoot;
    result->mScores += player.PlayGames( job.mPosition, count );
}
