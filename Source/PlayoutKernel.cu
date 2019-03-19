// PlayoutKernel.cu - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "Core.h"
#include "PlayoutJob.h"


__global__ void PlayGamesCuda( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< u64 > player( &job->mOptions, job->mRandomSeed );

    result->mScores = player.PlayGames( job->mPosition, count );
    result->mPathFromRoot = job->mPathFromRoot;
}


void QueuePlayGamesCuda( CudaLaunchSlot* slot, int blockCount, int blockSize )
{
    cudaEventRecord( slot->mStartEvent, slot->mStream );

    // Copy the inputs to device

    cudaMemcpyAsync( 
        slot->mInputDev, 
        slot->mInputHost, 
        sizeof( PlayoutJob ), 
        cudaMemcpyHostToDevice, 
        slot->mStream );

    // Queue the playout kernel

    PlayGamesCuda<<< blockCount, blockSize, 0, slot->mStream >>>( 
        slot->mInputDev, 
        slot->mOutputDev, 
        slot->mInfo.mNumGames );

    // Copy the results back to host

    cudaMemcpyAsync( 
        slot->mOutputHost, 
        slot->mOutputDev, 
        sizeof( PlayoutResult ), 
        cudaMemcpyDeviceToHost, 
        slot->mStream );

    cudaEventRecord( slot->mEndEvent, slot->mStream );
}

