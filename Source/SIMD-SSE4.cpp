// CPU-SSE4.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "PlayoutJob.h"

#if ENABLE_SSE4
#include "SIMD-SSE4.h"

CDECL ScoreCard PlayGamesSSE4( const PlayoutJob& job, int simdCount )
{
    GamePlayer< simd2_sse4 > player( &job.mOptions, job.mRandomSeed );
    return( player.PlayGames( job.mPosition, simdCount ) );
}
#endif
