// CPU-SSE2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#if ENABLE_SSE2
#include "SIMD-SSE2.h"

CDECL ScoreCard PlayGamesSSE2( const PlayoutOptions& options, const Position& pos, int simdCount )
{
    GamePlayer< simd2_sse2 > player( &options );
    return( player.PlayGames( pos, simdCount ) );
}
#endif 
