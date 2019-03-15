// CPU-AVX512.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"

#if ENABLE_AVX512
#include "SIMD-AVX512.h"

CDECL ScoreCard PlayGamesAVX512( const PlayoutOptions& options, const Position& pos, int simdCount )
{
    GamePlayer< simd8_avx512 > player( &options );
    return( player.PlayGames( pos, simdCount ) );
}
#endif 
