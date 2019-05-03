// GPU-CUDA.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_GPU_H__
#define CORVID_GPU_H__

#if SUPPORT_CUDA


#if RUNNING_ON_CUDA_HOST

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


class GpuWorker : public IAsyncWorker
{
    typedef std::map< cudaStream_t, std::list< CudaLaunchSlot* > > LaunchSlotsByStream;

    int                             mDeviceIndex;      
    std::unique_ptr< LaunchThread > mLaunchThread;
    std::unique_ptr< std::thread >  mJobThread;
    std::vector< CudaLaunchSlot >   mSlotInfo;      
    const GlobalOptions*            mOptions;

    Mutex                           mSlotMutex;
    std::vector< CudaLaunchSlot* >  mFreeSlots;
    LaunchSlotsByStream             mActiveSlotsByStream;


    bool mInitialized;
    PlayoutJobQueue*    mJobQueue;
    PlayoutResultQueue* mResultQueue;

public:    
    GpuWorker( const GlobalOptions* options, PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
    {
        mOptions = options;
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;
        mInitialized = false;
    }

    ~GpuWorker()
    {
        if( mInitialized )
            this->Shutdown();
    }

    static int GetDeviceCount()
    {
        int count;
        auto res = cudaGetDeviceCount( &count );
        if( res != cudaSuccess )
            count = 0;

        return( count );
    }

    void Initialize( int deviceIndex, int jobSlots )
    {
        mDeviceIndex  = deviceIndex;
        cudaSetDevice( deviceIndex );

        mLaunchThread = std::unique_ptr< LaunchThread >( new LaunchThread( mOptions, deviceIndex ) );
        mLaunchThread->Init();

        mSlotInfo.resize( jobSlots );

        for( int i = 0; i < jobSlots; i++ )
        {
            CudaLaunchSlot& slot  = mSlotInfo[i];

            slot.mStream       = (cudaStream_t) 0;
            slot.mInputHost    = mLaunchThread->mInputHost  + i;
            slot.mOutputHost   = mLaunchThread->mOutputHost + i;
            slot.mInputDev     = mLaunchThread->mInputDev   + i;
            slot.mOutputDev    = mLaunchThread->mOutputDev  + i;
            slot.mTickQueued   = 0;

            CUDA_REQUIRE(( cudaEventCreate( &slot.mStartEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &slot.mEndEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &slot.mReadyEvent ) ));

            mFreeSlots.push_back( &slot );
        }

        mJobThread = std::unique_ptr< std::thread >( new std::thread( [this] { this->RunJobThread(); } ) );

        mInitialized = true;
    }


    void Shutdown()
    {
        mLaunchThread.release();
        mJobThread->join();

        mInitialized = false;
    }

private:

    void RunJobThread()
    {
        for( ;; )
        {
            mSlotMutex.Enter();
            bool empty = mFreeSlots.empty();
            mSlotMutex.Leave();

            if( empty )
            {
                PlatSleep( 100 );
                continue;
            }

            PlayoutJobRef job = mJobQueue->Pop();
            if( job == NULL )
                return;

            MUTEX_SCOPE( mSlotMutex );

            assert( !mFreeSlots.empty() );
            CudaLaunchSlot* slot = mFreeSlots.back();
            mFreeSlots.pop_back();

            *slot->mInputHost = *job;
            slot->mTickQueued = PlatGetClockTick();

            //printf("Launching %p\n", slot );
            mLaunchThread->Launch( slot );
            //printf("Done\n" );

            mActiveSlotsByStream[slot->mStream].push_back( slot );
        }
    }

    virtual void Update() override
    {
        MUTEX_SCOPE( mSlotMutex );

        for( auto& kv : mActiveSlotsByStream )
        {
            auto& activeList = kv.second;

            while( !activeList.empty() )
            {
                CudaLaunchSlot* slot = activeList.front();
                if( cudaEventQuery( slot->mReadyEvent ) != cudaSuccess )
                    break;

                u64 tickReturned = PlatGetClockTick();

                activeList.pop_front();

                PlayoutResultRef result = PlayoutResultRef( new PlayoutResult() );
                *result = *slot->mOutputHost;

                cudaEventElapsedTime( &result->mGpuTime, slot->mStartEvent, slot->mEndEvent );
                result->mCpuLatency = (tickReturned - slot->mTickQueued) * 1000.0f / PlatGetClockFrequency();  

                mResultQueue->Push( result );

                //cudaEventDestroy( slot->mStartEvent );
                //cudaEventDestroy( slot->mEndEvent );
                //cudaEventDestroy( slot->mReadyEvent );

                /*
                printf("%d %d %d %.2f/%.2f %s\n", 
                    result->mScores.mWins[0], result->mScores.mWins[1], result->mScores.mPlays, 
                    result->mGpuTime, result->mCpuLatency, 
                    SerializeMoveList( result->mPathFromRoot ).c_str() );
                    */

                mFreeSlots.push_back( slot );
            }
        }
    }
};
  
#endif // RUNNING_ON_CUDA_HOST
#endif // SUPPORT_CUDA
#endif // CORVID_GPU_H__

