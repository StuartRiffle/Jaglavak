// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "SSE4.h"
#include "PlayoutBatch.h"
#include "GamePlayer.h"

extern void PlayGamesSSE4( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount )
{
    GamePlayer< simd2_sse4 > player( params );
    player.PlayGames( pos, dest, simdCount );
}
