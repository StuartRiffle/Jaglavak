// CUDA.cu - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "Job.h"
#include "CUDA.h"

__global__ void PlayGamesOnDevice( const PlayoutJobInfo* input, PlayoutJobresult* result, int count )
{
    GamePlayer player( &input->mOptions );

    result->mScores = player.PlayGames( input->mPos, count );
    result->mPathFromRoot = input->mPathFromRoot;
}


void QueuePlayoutJobCuda( CudaLaunchSlot* slot, int blockCount, int blockSize )
{
    cudaEventRecord( slot->mStartEvent, slot->mStream );

    // Copy the inputs to device

    cudaMemcpyAsync( 
        slot->mInputDev, 
        slot->mInputHost, 
        sizeof( PlayoutJobInfo ), 
        cudaMemcpyHostToDevice, 
        slot->mStream );

    // Queue the playout kernel

    PlayGamesOnDevice<<< blockCount, blockSize, 0, slot->mStream >>>( 
        slot->mInputDev, 
        slot->mOutputDev, 
        slot->mCount );

    cudaEventRecord( slot->mEndEvent, slot->mStream );

    // Copy the results back to host

    cudaMemcpyAsync( 
        slot->mOutputHost, 
        slot->mOutputDev, 
        sizeof( PlayoutJobResult ), 
        cudaMemcpyDeviceToHost, 
        slot->mStream );

    cudaEventRecord( job->mStopEvent, job->mStream );
}

