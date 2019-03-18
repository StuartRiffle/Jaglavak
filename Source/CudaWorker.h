// GPU-CUDA.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_GPU_H__
#define CORVID_GPU_H__

#if ENABLE_CUDA

struct CudaLaunchSlot
{
    PlayoutJob      mInfo;
    PlayoutResult    mResult;
    cudaStream_t        mStream;                    /// Stream this job was issued into
    cudaEvent_t         mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t         mEndEvent;                  /// GPU timer event to notify that the results have been copied back to host memory

    PlayoutJob*     mInputHost;                 /// Job input buffer, host side
    PlayoutResult*   mOutputHost;                /// Job output buffer, host side

    PlayoutJob*     mInputDev;                  /// Job input buffer, device side
    PlayoutResult*   mOutputDev;                 /// Job output buffer, device side

    u64                 mTickQueued;                /// CPU tick when the job was queued for execution
    u64                 mTickReturned;              /// CPU tick when the completed job was found
    float               mCpuLatency;                /// CPU time elapsed (in ms) between those two ticks, represents job processing latency
    float               mGpuTime;                   /// GPU time spent executing kernel (in ms)
};


#if CORVID_CUDA_HOST

#define CUDA_REQUIRE( _CALL ) \
{ \
    cudaError_t status = (_CALL); \
    if( status != cudaSuccess ) \
    { \
        fprintf( stderr, "ERROR: failure in " #_CALL " [%d]\n", status ); \
        return; \
    } \
}

struct LaunchThread
{
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties

    vector< cudaStream_t >      mStreamId;
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a job
    unique_ptr< std::thread >   mLaunchThread;
    ThreadSafeQueue< CudaLaunchSlot* > mLaunchQueue;

    PlayoutJob*             mInputHost;          /// Host-side job input buffer
    PlayoutResult*           mOutputHost;         /// Host-side job output buffer

    PlayoutJob*             mInputDev;          /// Device-side job input buffer
    PlayoutResult*           mOutputDev;         /// Device-side job output buffer

    Semaphore                   mThreadExited;

    LaunchThread( int deviceIndex, PlayoutJobQueue* jobQueue )
    {
        mDeviceIndex = deviceIndex;
        mJobQueue = jobQueue;
        mLaunchThread = new std::thread( [] { this->RunLaunchThread(); } );
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

        slot->mStreamIndex = mStreamIndex;
        slot->mStreamId = mStreamId[mStreamIndex];

        mLaunchQueue.Push( slot );
    }

    void InitCuda()
    {
        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));

        for( int i = 0; i < CUDA_STREAM_COUNT; i++ )
        {
            cudaStream_t stream;
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( &stream, cudaStreamNonBlocking ) ));

            mStreamId[i] = stream;
        }

        size_t inputBufSize     = mJobSlots * sizeof( PlayoutJob );
        size_t outputBufSize    = mJobSlots * sizeof( PlayoutResult );

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

        for( size_t i = 0; i < CUDA_STREAM_COUNT; i++ )
            cudaStreamDestroy( mStreamId[i] );
    }

    void RunLaunchThread()
    {
        this->InitCuda();

        for( ;; )
        {
            CudaLaunchSlot* slot = mLaunchQueue.Pop();
            if( slot == NULL )
                break;

            extern void QueuePlayoutJobCuda( CudaLaunchSlot* slot, int blockCount, int blockSize );
            QueuePlayoutJobCuda( slot, blockCount, blockSize );
        }

        this->ShutdownCuda();

        mThreadExited->Post();
    }
};


class CudaAsyncWorker : public IAsyncWorker
{
    typedef std::map< int, std::list< CudaLaunchSlot* > > LaunchSlotsByStream;

    int                             mDeviceIndex;      
    std::unique_ptr< LaunchThread > mLaunchThread;
    std::unique_ptr< std::thread >  mJobThread;
    std::vector< CudaLaunchSlot >   mSlotInfo;      
    std::vector< CudaLaunchSlot* >  mFreeSlots;
    LaunchSlotsByStream             mActiveSlotsByStream;

    bool mInitialized;
    PlayoutJobQueue*    mJobQueue;
    PlayoutResultQueue* resultQueue;

    
    CudaAsyncWorker( PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
    {
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;
        mInitialized = false;
    }

    ~CudaAsyncWorker()
    {
        if( mInitialized )
            this->Shutdown();
    }

    void Initialize( int deviceIndex, int jobSlots )
    {
        mDeviceIndex  = deviceIndex;
        cudaSetDevice( deviceIndex );

        mLaunchThread = new LaunchThread( deviceIndex );

        mSlotInfo.resize( jobSlots );

        for( int i = 0; i < jobSlots; i++ )
        {
            CudaLaunchSlot& slot  = mSlotInfo[i];

            slot.mDevice       = this;
            slot.mStream       = (cudaStream_t) 0;
            slot.mInputHost    = mLaunchThread->mInputHost  + i;
            slot.mOutputHost   = mLaunchThread->mOutputHost + i;
            slot.mInputDev     = mLaunchThread->mInputDev   + i;
            slot.mOutputDev    = mLaunchThread->mOutputDev  + i;
            slot.mInitialized  = false;
            slot.mTickQueued   = 0;
            slot.mTickReturned = 0;
            slot.mCpuLatency   = 0;
            slot.mGpuTime      = 0;

            CUDA_REQUIRE(( cudaEventCreate( &slot.mStartEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &slot.mEndEvent ) ));

            mFreeSlots.push_back( &slot );
        }

        mJobThread = new std::thread( [] { this->RunJobThread(); } );

        mInitialized = true;
    }


    void Shutdown()
    {
        mLaunchThread.release();
        mJobThread.join();

        cudaSetDevice( mDeviceIndex );

        for( auto& job : mSlotInfo )
        {
            cudaEventDestroy( job.mStartEvent );
            cudaEventDestroy( job.mEndEvent );
            cudaEventDestroy( job.mReadyEvent );
        }

        mInitialized = false;
    }

    void RunJobThread()
    {
        for( ;; )
        {
            std::vector< PlayoutJobRef > jobs = mJobQueue->PopMultiple( mFreeSlots.size() );
            for( auto& job : jobs )
            {
                if( job == NULL )
                    return;

                assert( !mFreeSlots.empty() );
                CudaLaunchSlot* slot = mFreeSlots.back();
                mFreeSlots.pop_back();

                slot->mInfo = *job;

                mLaunchThread->Launch( slot );

                slot->mTickQueued = PlatGetClockTick();

                mActiveSlots[slot->mStreamIndex].push_back( slot );
            }
        }
    }

    override void Update()
    {
        std::vector< PlayoutResultRef > completed;

        for( auto& kv : mActiveSlotsByStream )
        {
            auto& activeList = kv.second;

            while( !activeList.empty() )
            {
                CudaLaunchSlot* slot = activeList.front();
                if( cudaEventQuery( slot->mReadyEvent ) != cudaSuccess )
                    break;

                activeList.pop_front();

                PlayoutResultRef result = new PlayoutResult();
                *result = slot->mResult;

                completed.push_back( result );
            }
        }

        mResultQueue->Push( completed.data(), completed.size() );
    }
}
  








#endif // CORVID_CUDA_HOST
#endif // ENABLE_CUDA
#endif // CORVID_GPU_H__

