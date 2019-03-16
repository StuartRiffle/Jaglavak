// CUDA.cu - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "Job.h"
#include "CUDA.h"

__global__ void PlayGamesOnDevice( const PlayoutJobInfo* input, PlayoutJobresult* result, int count )
{
    GamePlayer player( &input->mOptions );

    result.mScores = player.PlayGames( input->mPos, count );
    result.mPathFromRoot = input->mPathFromRoot;
}

void QueuePlayoutJobCuda( CudaJob* job, int blockCount, int blockSize )
{
    // Copy the inputs to device

    cudaMemcpyAsync( 
        job->mInputDev, 
        job->mInputHost, 
        sizeof( PlayoutJobInfo ), 
        cudaMemcpyHostToDevice, 
        job->mStream );

    // Queue the playout kernel

    cudaEventRecord( job->mStartEvent, job->mStream );

    PlayGamesOnDevice<<< blockCount, blockSize, 0, job->mStream >>>( 
        job->mInputDev, 
        job->mOutputDev, 
        job->mCount );

    cudaEventRecord( job->mEndEvent, job->mStream );

    // Copy the results back to host

    cudaMemcpyAsync( 
        job->mOutputHost, 
        job->mOutputDev, 
        sizeof( PlayoutJobResult ), 
        cudaMemcpyDeviceToHost, 
        job->mStream );

    // Record an event we can test for completion

    cudaEventRecord( job->mReadyEvent, job->mStream );
}

