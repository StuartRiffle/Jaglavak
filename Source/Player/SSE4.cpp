// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#define ENABLE_POPCNT (1)

#include "Platform.h"
#include "Chess.h"
#include "ScoreCard.h"
#include "GamePlayer.h"
#include "SSE4.h"

extern void PlayGamesSSE4( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd2_sse4 > player( params );
    player.PlayGames( pos, dest, simdCount );
}
