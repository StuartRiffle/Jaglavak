// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "PlayoutBatch.h"
#include "GamePlayer.h"

__global__ void PlayGamesCuda( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    GamePlayer player( params, x );
    
    int idx = x % count;

    ScoreCard scores;
    player.PlayGames( pos + idx, &scores, 1 );

    atomicAdd( (unsigned long long*) &dest[idx].mWins[BLACK], scores.mWins[BLACK] );
    atomicAdd( (unsigned long long*) &dest[idx].mWins[WHITE], scores.mWins[WHITE] );
    atomicAdd( (unsigned long long*) &dest[idx].mPlays, scores.mPlays );
}

void PlayGamesCudaAsync( CudaLaunchSlot* slot, int blockCount, int blockSize, cudaStream_t stream )
{
    PlayGamesCuda<<< blockCount, blockSize, 0, stream >>>(
        slot->mParams.mDev,
        slot->mInputs.mDev, 
        slot->mOutputs.mDev, 
        slot->mCount );
}
