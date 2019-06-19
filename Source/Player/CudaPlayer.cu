// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "ScoreCard.h"
#include "GamePlayer.h"

__global__ void PlayGamesCuda( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx = x % count;
    
    int salt = x;
    GamePlayer< u64 > player( params + idx, salt );

    player.PlayGames( pos + idx, dest + idx, 1 );
}

void PlayGamesCudaAsync( 
    const PlayoutParams* params, 
    const Position* pos, 
    ScoreCard* dest, 
    int count,
    int blockCount, 
    int blockSize, 
    cudaStream_t stream )
{
    PlayGamesCuda<<< blockCount, blockSize, 0, stream >>>( params, pos, dest, count );
}
