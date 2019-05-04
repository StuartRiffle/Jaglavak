// PlayoutKernel.cu - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "GamePlayer.h"
#include "PlayoutJob.h"

__global__ void PlayGamesCuda( const PlayoutJob* job, PlayoutResult* result, int count )
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    GamePlayer< u64 > player( &job->mOptions, job->mRandomSeed + idx );

    result->mPathFromRoot = job->mPathFromRoot;

    ScoreCard scores = player.PlayGames( job->mPosition, count );

    atomicAdd( (unsigned long long*) &result->mScores.mWins[BLACK], scores.mWins[BLACK] );
    atomicAdd( (unsigned long long*) &result->mScores.mWins[WHITE], scores.mWins[WHITE] );
    atomicAdd( (unsigned long long*) &result->mScores.mPlays, scores.mPlays );
}


void QueuePlayGamesCuda( CudaLaunchSlot* slot, int blockCount, int blockSize )
{
    // Clear the output buffer

    cudaMemsetAsync( 
        slot->mOutputDev,
        0,
        sizeof( PlayoutResult ),
        slot->mStream );

    // Copy the inputs to device

    cudaMemcpyAsync( 
        slot->mInputDev, 
        slot->mInputHost, 
        sizeof( PlayoutJob ), 
        cudaMemcpyHostToDevice, 
        slot->mStream );

    // Run the playout kernel

    //cudaEventCreate( &slot->mStartEvent );
    cudaEventRecord( slot->mStartEvent, slot->mStream );

    PlayGamesCuda<<< blockCount, blockSize, 0, slot->mStream >>>( 
        slot->mInputDev, 
        slot->mOutputDev, 
        1 );

    // Copy the results back to host

    cudaMemcpyAsync( 
        slot->mOutputHost, 
        slot->mOutputDev, 
        sizeof( PlayoutResult ), 
        cudaMemcpyDeviceToHost, 
        slot->mStream );

    cudaEventRecord( slot->mReadyEvent, slot->mStream );
}
