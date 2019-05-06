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
    int mCount;

    Position mPosition[PLAYOUT_BATCH_MAX];

    // This gets carried along so we know where the results should go

    ScoreCard mResults[PLAYOUT_BATCH_MAX];

    // Outputs

    MoveList mPathFromRoot[PLAYOUT_BATCH_MAX];

    PlayoutBatch() : mCount( 0 ) {}

    void Append( const Position& pos, const MoveList& pathFromRoot )
    {
        assert( mCount < PLAYOUT_BATCH_MAX );

        mPosition[mCount] == pos;
        mPathFromRoot[mCount] = pathFromRoot;
        mCount++;
    }
};

typedef std::shared_ptr< PlayoutBatch > BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

template< typename T >
struct CudaBuffer
{
    T* mHost;
    T* mDevice;
    size_t mBufferSize;

    CudaBuffer() : mHost( NULL ), mDevice( NULL ), mBufferSize( 0 ) {}

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

    void Init( int count )
    {
        mHost = NULL;
        mDevice = NULL;
        mBufferSize = count * sizeof( T );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mHost, mBufferSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mDevice, mBufferSize ) ));        
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
    BatchRef                    mBatch;
    CudaBuffer< PlayoutParams > mParams;
    CudaBuffer< Position >      mInputs;
    CudaBuffer< ScoreCard >     mOutputs;
    int                         mCount;
    cudaEvent_t                 mReadyEvent; 
};

