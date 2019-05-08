// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Threads.h"
#include "GamePlayer.h"

extern void PlayGamesX64( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    GamePlayer< u64 > player( params );
    player.PlayGames( pos, dest, count );
}

