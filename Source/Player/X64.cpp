// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "PlayoutBatch.h"
#include "GamePlayer.h"

extern _CDECL void PlayGamesX64( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    GamePlayer player( params );
    player.PlayGames( pos, dest, count );
}

