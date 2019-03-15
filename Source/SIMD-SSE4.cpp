// CPU-SSE4.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#if ENABLE_SSE4
#include "SIMD-SSE4.h"

CDECL ScoreCard PlayGamesSSE4( const PlayoutOptions& options, const Position& pos, int simdCount )
{
    GamePlayer< simd2_sse4 > player( &options );
    return( player.PlayGames( pos, simdCount ) );
}
#endif
