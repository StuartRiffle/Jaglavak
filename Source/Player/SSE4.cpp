// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

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
