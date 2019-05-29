// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#define ENABLE_POPCNT (1)

#include "Platform.h"
#include "Chess.h"
#include "ScoreCard.h"
#include "GamePlayer.h"
#include "AVX512.h"

extern void PlayGamesAVX512( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd8_avx512 > player( params );
    player.PlayGames( pos, dest, simdCount );
}
