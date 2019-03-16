// GPU-CUDA.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_GPU_H__
#define CORVID_GPU_H__

#if ENABLE_CUDA


struct CudaJobMetrics
{
    u64                 mTickQueued;                /// CPU tick when the job was queued for execution
    u64                 mTickReturned;              /// CPU tick when the completed job was found
    float               mCpuLatency;                /// CPU time elapsed (in ms) between those two ticks, represents job processing latency
    float               mGpuTime;                   /// GPU time spent executing kernel (in ms)
};

struct CudaLaunchSlot
{
    PlayoutJobInfo      mInfo;
    PlayoutJobResult    mResult;
    cudaStream_t        mStream;                    /// Stream this job was issued into
    cudaEvent_t         mStartEvent;                /// GPU timer event to mark the start of kernel execution
    cudaEvent_t         mEndEvent;                  /// GPU timer event to mark the end of kernel execution
    cudaEvent_t         mReadyEvent;                /// GPU timer event to notify that the results have been copied back to host memory

    PlayoutJobInfo*     mInputHost;                 /// Job input buffer, host side
    PlayoutJobResult*   mOutputHost;                /// Job output buffer, host side

    PlayoutJobInfo*     mInputDev;                  /// Job input buffer, device side
    PlayoutJobResult*   mOutputDev;                 /// Job output buffer, device side
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



class CudaAsyncWorker : public IAsyncWorker
{
    int                             mDeviceIndex;      
    std::unique_ptr< CudaThread >   mCudaThread;
    std::vector< CudaLaunchSlot >      mJobInfo;      
    std::list< CudaLaunchSlot* >       mActiveSlots;
    std::list< CudaLaunchSlot* >       mFreeSlots;
    
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties

    ThreadSafeQueue< CudaLaunchSlot* > mLaunchQueue;
    cudaStream_t                mStreamId[CUDA_STREAM_COUNT];
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a job
    unique_ptr< std::thread >   mLaunchThread;

    PlayoutJobInfo*             mInputHost;          /// Host-side job input buffer
    PlayoutJobResult*           mOutputHost;         /// Host-side job output buffer

    PlayoutJobInfo*             mInputDev;          /// Device-side job input buffer
    PlayoutJobResult*           mOutputDev;         /// Device-side job output buffer

    Semaphore                   mThreadStarted;
    Semaphore                   mThreadExited;

    Cuda

    ~CudaAsyncWorker()
    {
        this->Shutdown();
    }


    void Initialize( int deviceIndex, int jobSlots )
    {
        mDeviceIndex  = deviceIndex;
        mCudaThread   = new CudaThread( deviceIndex );

        cudaSetDevice( deviceIndex );

        mJobInfo.resize( jobSlots );

        for( int i = 0; i < jobSlots; i++ )
        {
            CudaLaunchSlot& job  = mJobInfo[i];

            job.mDevice       = this;
            job.mStream       = (cudaStream_t) 0;
            job.mInputHost    = mCudaThread->mInputHost  + i;
            job.mOutputHost   = mCudaThread->mOutputHost + i;
            job.mInputDev     = mCudaThread->mInputDev   + i;
            job.mOutputDev    = mCudaThread->mOutputDev  + i;
            job.mInitialized  = false;
            job.mTickQueued   = 0;
            job.mTickReturned = 0;
            job.mCpuLatency   = 0;
            job.mGpuTime      = 0;

            CUDA_REQUIRE(( cudaEventCreate( &job.mStartEvent ) ));
            CUDA_REQUIRE(( cudaEventCreate( &job.mEndEvent ) ));
            CUDA_REQUIRE(( cudaEventCreateWithFlags( &job.mReadyEvent, cudaEventDisableTiming ) ));

            mFreeSlots.push( &job );
        }
    }


    void Shutdown()
    {
        mCudaThread.release();

        for( auto& job : mJobInfo )
        {
            cudaEventDestroy( job.mStartEvent );
            cudaEventDestroy( job.mEndEvent );
            cudaEventDestroy( job.mReadyEvent );
        }
    }


    void JobThread()
    {
        char threadName[80];
        sprintf( threadName, "CUDA %d", mDeviceIndex );
        PlatSetThreadName( threadName );

        for( ;; )
        {
            vector< PlayoutJobRef > jobs = mIncoming->PopMultiple( mFreeSlots.size() );
            for( auto& job : jobs )
            {
                if( job == NULL )
                    break;

                CudaLaunchSlot* slot = mFreeSlots.pop();
                slot->mInfo = *job;

                mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
                slot->mStream = mStreamId[mStreamIndex];

                extern void QueuePlayoutJobCuda( CudaLaunchSlot* slot, int blockCount, int blockSize );
                QueuePlayoutJobCuda( slot, blockCount, blockSize );

                slot->mTickQueued = PlatGetClockTick();
                mActiveSlots.push_back( slot );
            }
        }
    }

    override void Update()
    {
        vector< PlayoutJobResultRef > done;
        done.reserve( mActiveSlots.size() );

        while( !mActiveSlots.empty() )
        {
            CudaLaunchSlot* slot = mActiveSlots.front();
            if( cudaEventQuery( slot->mReadyEvent ) != cudaSuccess )
                break;

            mActiveSlots.pop_front();

            PlayoutJobResultRef result = new PlayoutJobResult();
            *result = slot->mResult;

            done.push_back( result );
        }

        mResultQueue->Push( done.data(), done.size() );
    }
}
  


struct CudaThread
{
    CudaThread( int deviceIndex )
    {
        mDeviceIndex = deviceIndex;

        mLaunchThread = new std::thread( [] { this->LaunchThread(); } );
        mThreadStarted.Wait();
    }

    ~CudaThread()
    {
        mJobQueue.Push( NULL );
        mThreadExited.Wait();

        mLaunchThread.join();

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

        size_t inputBufSize     = mJobSlots * sizeof( PlayoutJobInfo );
        size_t outputBufSize    = mJobSlots * sizeof( PlayoutJobResult );

        CUDA_REQUIRE(( cudaMallocHost( (void**) &mInputHost,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMallocHost( (void**) &mOutputHost, outputBufSize ) ));

        CUDA_REQUIRE(( cudaMalloc( (void**) &mInputDev,  inputBufSize ) ));
        CUDA_REQUIRE(( cudaMalloc( (void**) &mOutputDev, outputBufSize ) ));

        CUDA_REQUIRE(( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) ));
    }



    void LaunchThread()
    {
        this->InitCuda();

        mThreadStarted.Post();

        int warpSize = mProp.warpSize;

        for( ;; )
        {
            CudaLaunchSlot* slot = mLaunchQueue.Pop();
            if( slot == NULL )
                break;

            mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
            slot->mStream = mStreamId[mStreamIndex];

            extern void QueuePlayoutJobCuda( CudaLaunchSlot* slot, int blockCount, int blockSize );

            QueuePlayoutJobCuda( slot, blockCount, blockSize );
            slot->mTickQueued = PlatGetClockTick();
        }

        mThreadExited.Post();
    }


};






#endif // CORVID_CUDA_HOST
#endif // ENABLE_CUDA
#endif // CORVID_GPU_H__

