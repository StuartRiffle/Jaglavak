// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once


struct PlayoutParams
{
    u64     mRandomSeed;
    int     mNumGamesEach;
    int     mMaxMovesPerGame;
};

struct PlayoutBatch
{
    // Inputs

    PlayoutParams mParams;
    std::vector< Position > mPosition;

    // Outputs

    std::vector< MoveList > mPathFromRoot;

    // This gets carried along so we know where the results should go

    std::vector< ScoreCard > mResults;
};

typedef std::shared_ptr< PlayoutBatch > BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

template< typename T >
struct CudaBuffer
{
    T* mHost;
    T* mDevice;
    size_t mBufferSize;

    CudaBuffer( int count )
    {
        mHost = NULL;
        mDevice = NULL;
        mBufferSize = count * sizeof( T );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mHost, mBufferSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mDevice, mBufferSize ) ));        
    }

    ~CudaBuffer()
    {
        if( mHost )
            cudaFreeHost( mHost );

        if( mDevice )
            cudaFree( mDevice );
    }

    T& operator[]( size_t idx )
    {
        return mHost[idx];
    }

    void CopyToDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mDevice, mHost, mBufferSize, cudaMemcpyHostToDevice, stream );
    }

    void CopyToHostAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemcpyAsync( mHost, mDevice, mBufferSize, cudaMemcpyDeviceToHost, stream );
    }

    void ClearOnDeviceAsync( cudaStream_t stream = NULL ) const
    {
        cudaMemsetAsync( mDevice, 0, mBufferSize, stream );
    }   
};

struct CudaLaunchSlot
{
    CudaBuffer< PlayoutParams > mParams;
    CudaBuffer< Position >      mInputs;
    CudaBuffer< ScoreCard >     mOutputs;
    int                         mCount;
    cudaEvent_t                 mReadyEvent; 
};

