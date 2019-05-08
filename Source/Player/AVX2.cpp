// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "AVX2.h"
#include "PlayoutBatch.h"
#include "GamePlayer.h"

extern void PlayGamesAVX2( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd4_avx2 > player( params );
    player.PlayGames( pos, dest, simdCount );
}