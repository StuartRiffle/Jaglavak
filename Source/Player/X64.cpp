// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#define ENABLE_POPCNT (0)

#include "Platform.h"
#include "Chess/Core.h"
#include "ScoreCard.h"
#include "GamePlayer.h"

extern void PlayGamesX64( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    GamePlayer< u64 > player( params );
    player.PlayGames( pos, dest, count );
}

