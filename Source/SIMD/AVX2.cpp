// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#define ENABLE_POPCNT (1)

#include "Platform.h"
#include "Chess.h"
#include "ScoreCard.h"
#include "GamePlayer.h"
#include "AVX2.h"

extern void PlayGamesAVX2( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd4_avx2 > player( params );
    player.PlayGames( pos, dest, simdCount );
}