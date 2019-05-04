#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        printf( "ERROR: failure in " #_CALL " [%d]\n", status ); \
        return; \
    } \
}

struct LaunchThread
{
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties
    const GlobalOptions*        mOptions;

    std::vector< cudaStream_t >      mStreamId;
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a job
    std::unique_ptr< std::thread >   mLaunchThread;
    ThreadSafeQueue< CudaLaunchSlot* > mLaunchQueue;

    PlayoutJob*             mInputHost;          /// Host-side job input buffer
    PlayoutResult*           mOutputHost;         /// Host-side job output buffer

    PlayoutJob*             mInputDev;          /// Device-side job input buffer
    PlayoutResult*           mOutputDev;         /// Device-side job output buffer

    Semaphore                   mThreadRunning;
    Semaphore                   mThreadExited;

    LaunchThread( const GlobalOptions* options, int deviceIndex )
    {
        mOptions = options;
        mDeviceIndex = deviceIndex;
    }

    void Init()
    {
        mLaunchThread = std::unique_ptr< std::thread >( new std::thread( [this] { this->RunLaunchThread(); } ) );

        mThreadRunning.Wait();
    }

    ~LaunchThread()
    {
        mLaunchQueue.Push( NULL );
        mThreadExited.Wait();
    }

    void Launch( CudaLaunchSlot* slot )
    {
        mStreamIndex++;
        mStreamIndex %= mStreamId.size();

        slot->mStream = mStreamId[mStreamIndex];

        mLaunchQueue.Push( slot );
    }

    void InitCuda()
    {
        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));


        int coresPerSM = 0;
        switch( mProp.major )
        {
            case 2:     coresPerSM = (mProp.minor > 0)? 48 : 32; break; // Fermi
            case 3:     coresPerSM = 192; break; // Kepler
            case 5:     coresPerSM = 128; break; // Maxwell
            case 6:     coresPerSM = (mProp.minor > 0)? 128 : 64; break; // Pascal
            default:    coresPerSM = 64; break; // Volta+
        }

        int totalCores = mProp.multiProcessorCount * coresPerSM;

        printf( "[GPU %d] %5d %4d %s\n", mDeviceIndex, mProp.multiProcessorCount, coresPerSM, mProp.name );

        for( int i = 0; i < mOptions->mCudaStreams; i++ )
        {
            cudaStream_t stream;
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) ));

            mStreamId.push_back( stream );
        }

        size_t inputBufSize     = mOptions->mCudaQueueDepth * sizeof( PlayoutJob );
        size_t outputBufSize    = mOptions->mCudaQueueDepth * sizeof( PlayoutResult );

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

        for( auto& stream : mStreamId )
            cudaStreamDestroy( stream );
    }

    void RunLaunchThread()
    {
        this->InitCuda();

        mThreadRunning.Post();

        for( ;; )
        {
            CudaLaunchSlot* slot = mLaunchQueue.Pop();
            if( slot == NULL )
                break;

            int blockCount = (slot->mInputHost->mNumGames + mProp.warpSize - 1) / mProp.warpSize;

            extern void QueuePlayGamesCuda( CudaLaunchSlot* slot, int blockCount, int blockSize );
            QueuePlayGamesCuda( slot, blockCount, mProp.warpSize );
        }

        this->ShutdownCuda();

        mThreadExited.Post();
    }
};
