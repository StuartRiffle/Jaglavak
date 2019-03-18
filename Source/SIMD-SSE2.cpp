// CPU-SSE2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "PlayoutJob.h"

#if ENABLE_SSE2
#include "SIMD-SSE2.h"

CDECL ScoreCard PlayGamesSSE2( const PlayoutJob& job, int simdCount )
{
    GamePlayer< simd2_sse2 > player( &job.mOptions, job.mRandomSeed );
    return( player.PlayGames( job.mPosition, simdCount ) );
}
#endif 
