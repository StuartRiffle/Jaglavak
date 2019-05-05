// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        printf( "ERROR: failure in " #_CALL " [%d]\n", status ); \
        return; \
    } \
}

enum
{
    CUDA_MAX_STREAMS = 16,
};

struct CudaLauncher
{
    int                     mDeviceIndex;       /// CUDA device index
    cudaDeviceProp          mProp;              /// CUDA device properties
    const GlobalOptions*    mOptions;

    cudaStream_t            mStreamId[CUDA_MAX_STREAMS];
    int                     mStreamIndex;       /// Index of the next stream to be used for submitting a job

    PlayoutBatch*             mInputHost;          /// Host-side job input buffer
    PlayoutResult*          mOutputHost;         /// Host-side job output buffer

    PlayoutBatch*             mInputDev;          /// Device-side job input buffer
    PlayoutResult*          mOutputDev;         /// Device-side job output buffer

    ThreadSafeQueue< CudaLaunchSlot* > mLaunchQueue;
    std::unique_ptr< std::thread > mLaunchThread;
    Semaphore mLaunchThreadRunning;

    void InitCuda()
    {
        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));

        for( int i = 0; i < CUDA_MAX_STREAMS; i++ )
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( mStreamId + i, cudaStreamNonBlocking ) ));

        size_t inputBufSize  = mOptions->mCudaQueueDepth * sizeof( PlayoutBatch );
        size_t outputBufSize = mOptions->mCudaQueueDepth * sizeof( PlayoutResult );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mInputHost,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMallocHost( (void**) &mOutputHost, outputBufSize ) ));

        CUDA_REQUIRE(( cudaMalloc( (void**) &mInputDev,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mOutputDev, outputBufSize ) ));

        CUDA_REQUIRE(( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) ));
    }

    void ShutdownCuda()
    {
        if( mInputHost )
            cudaFreeHost( mInputHost );

        if( mOutputHost )
            cudaFreeHost( mOutputHost );

        if( mInputDev )
            cudaFree( mInputDev );

        if( mOutputDev )
            cudaFree( mOutputDev );

        for( int i = 0; i < CUDA_MAX_STREAMS; i++ )
            cudaStreamDestroy( mStreamId[i] );
    }

    void LaunchThread()
    {
        this->InitCuda();
        mLaunchThreadRunning.Post();

        for( ;; )
        {
            CudaLaunchSlot* slot;
            if( !mLaunchQueue.Pop( slot ) )
                break;

            mStreamIndex++;
            mStreamIndex %= CUDA_MAX_STREAMS;

            int blockCount = (slot->mInputHost->mNumGames + mProp.warpSize - 1) / mProp.warpSize;

            extern void QueuePlayGamesCuda( CudaLaunchSlot* slot, int blockCount, int blockSize, cudaStream_t stream );
            QueuePlayGamesCuda( slot, blockCount, mProp.warpSize, mStreamId[mStreamIndex] );
        }

        this->ShutdownCuda();
    }

public:
    CudaLauncher( const GlobalOptions* options, int deviceIndex )
    {
        mOptions = options;
        mDeviceIndex = deviceIndex;
    }

    ~CudaLauncher()
    {
        mLaunchQueue.Terminate();

        if( mLaunchThread )
            mLaunchThread->join();
    }

    void Init()
    {
        mLaunchThread = std::unique_ptr< std::thread >( new std::thread( [this] { this->LaunchThread(); } ) );
        mLaunchThreadRunning.Wait();
    }

    void Launch( CudaLaunchSlot* slot )
    {
        mLaunchQueue.Push( slot );
    }
};
