// CPU-AVX2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#if ENABLE_AVX2
#include "SIMD-AVX2.h"

extern CDECL ScoreCard PlayGamesAVX2( const PlayoutOptions& options, const Position& pos, int simdCount )
{
    GamePlayer< simd4_avx2 > player( &options );
    return( player.PlayGames( pos, simdCount ) );
}
#endif 
