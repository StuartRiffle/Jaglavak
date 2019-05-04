// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "SSE4.h"
#include "PlayoutJob.h"
#include "GamePlayer.h"

extern _CDECL void PlayGamesSSE4( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< simd2_sse4 > player( &job.mOptions, job.mRandomSeed );

    result->mPathFromRoot = job->mPathFromRoot;
    result->mScores += player.PlayGames( job.mPosition, count );
}
