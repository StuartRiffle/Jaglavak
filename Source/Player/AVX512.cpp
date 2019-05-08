// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "GamePlayer.h"
#include "AVX512.h"

extern void PlayGamesAVX512( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd8_avx512 > player( params );
    player.PlayGames( pos, dest, simdCount );
}
