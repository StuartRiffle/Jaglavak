// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "PlayoutBatch.h"
#include "GamePlayer.h"

__global__ void PlayGamesCuda( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    idx %= count;

    GamePlayer player( params, idx );
    ScoreCard scores;

    player.PlayGames( pos + idx, &scores, 1 );

    atomicAdd( (unsigned long long*) &dest[idx].mScores.mWins[BLACK], scores.mWins[BLACK] );
    atomicAdd( (unsigned long long*) &dest[idx].mScores.mWins[WHITE], scores.mWins[WHITE] );
    atomicAdd( (unsigned long long*) &dest[idx].mScores.mPlays, scores.mPlays );
}

void PlayGamesCudaAsync( CudaLaunchSlot* slot, int blockCount, int blockSize, cudaStream_t stream )
{
    slot->mParams.CopyToDeviceAsync( stream );
    slot->mInputs.CopyToDeviceAsync( stream );
    slot->mOutput.ClearOnDevice( stream );

    PlayGamesCuda<<< blockCount, blockSize, 0, stream >>>(
        slot->mParams.mDev,
        slot->mInputs.mDev, 
        slot->mOutputs.mDev, 
        slot->mCount );

    slot->mOutputs.CopyToHostAsync( stream );
    cudaEventRecord( slot->mReadyEvent, stream );
}
