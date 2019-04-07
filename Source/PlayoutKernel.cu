// PlayoutKernel.cu - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "Core.h"
#include "PlayoutJob.h"

#if SUPPORT_CUDA

__global__ void PlayGamesCuda( const PlayoutJob* job, PlayoutResult* result, int count )
{
    GamePlayer< u64 > player( &job->mOptions, job->mRandomSeed );

    result->mScores = player.PlayGames( job->mPosition, count );
    result->mPathFromRoot = job->mPathFromRoot;
}


void QueuePlayGamesCuda( CudaLaunchSlot* slot, int blockSize )
{

    // Copy the inputs to device

    cudaMemcpyAsync( 
        slot->mInputDev, 
        slot->mInputHost, 
        sizeof( PlayoutJob ), 
        cudaMemcpyHostToDevice, 
        slot->mStream );

    // Run the playout kernel

    cudaEventRecord( slot->mStartEvent, slot->mStream );

    int blockCount = AlignUp( slot->mInfo.mNumGames, blockSize );
    PlayGamesCuda<<< blockCount, blockSize, 0, slot->mStream >>>( 
        slot->mInputDev, 
        slot->mOutputDev, 
        1 );

    cudaEventRecord( slot->mEndEvent, slot->mStream );

    // Copy the results back to host

    cudaMemcpyAsync( 
        slot->mOutputHost, 
        slot->mOutputDev, 
        sizeof( PlayoutResult ), 
        cudaMemcpyDeviceToHost, 
        slot->mStream );

    cudaEventRecord( slot->mReadyEvent, slot->mStream );
}

#endif // SUPPORT_CUDA
