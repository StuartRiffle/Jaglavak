// GPU-CUDA.cu - CORVID CHESS ENGINE (c) 2012-2016 Stuart Riffle

#include "Core.h"

__global__ void PlayGamesOnDevice( const PlayoutJobInput* input, PlayoutJobOutput* output, int count )
{
    PlayoutProvider player( &input->mOptions );

    output->mScores = player.PlayGames( input->mPos, count );
    output->mPathFromRoot = input->mPathFromRoot;
}

void QueuePlayoutJobCuda( PlayoutJobInput* batch, int blockCount, int blockSize, i32* exitFlag )
{
    // Copy the inputs to device

    cudaMemcpyAsync( 
        batch->mInputDev, 
        batch->mInputHost, 
        sizeof( PlayoutJobInput ) * batch->mCount, 
        cudaMemcpyHostToDevice, 
        batch->mStream );

    // Clear the device outputs

    cudaMemsetAsync( 
        batch->mOutputDev, 
        0, 
        sizeof( PlayoutJobOutput ) * batch->mCount, 
        batch->mStream );

    // Run the search kernel

    cudaEventRecord( batch->mStartEvent, batch->mStream );

    SearchPositionsOnGPU<<< blockCount, blockSize, 0, batch->mStream >>>( 
        batch->mInputDev, 
        batch->mOutputDev, 
        batch->mCount, 
        stride, 
        batch->mHashTableDev, 
        batch->mEvaluatorDev,
        batch->mOptionsDev,
        exitFlag );

    cudaEventRecord( batch->mEndEvent, batch->mStream );

    // Copy the outputs to host

    cudaMemcpyAsync( batch->mOutputHost, batch->mOutputDev, sizeof( PlayoutJobOutput ) * batch->mCount, cudaMemcpyDeviceToHost, batch->mStream );

    // Record an event we can test for completion

    cudaEventRecord( batch->mReadyEvent, batch->mStream );
}

