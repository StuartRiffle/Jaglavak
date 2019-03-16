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

struct CudaJobInfo
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
    int                         mDeviceIndex;      

    std::vector< CudaJobInfo >      mJobInfo;      
    std::list< CudaJobInfo* >       mRunningJobs;
    std::list< CudaJobInfo* >       mFreeSlots;
    std::unique_ptr< CudaThread >   mCudaThread;
    
    ~CudaAsyncWorker()
    {
        this->Shutdown();
    }


    void Initialize( int deviceIndex, int jobSlots )
    {
        mDeviceIndex  = deviceIndex;
        mCudaThread   = new CudaThread( deviceIndex );

        mJobInfo.resize( jobSlots );

        for( int i = 0; i < jobSlots; i++ )
        {
            CudaJobInfo& job  = mJobInfo[i];

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

    void Update()
    {
        vector< PlayoutJobResultRef > done;

        while( !mRunningJobs.empty() )
        {
            CudaJobInfo* job = mRunningJobs.front();
            if( cudaEventQuery( job->mReadyEvent ) != cudaSuccess )
                break;

            mRunningJobs.pop_front();

            PlayoutJobResultRef result = new PlayoutJobResult();
            done.emplace_back( job->mResult );
        }

        mResultQueue->Push( done.data(), done.size() );
    }
}
  


struct CudaThread
{
    int                         mDeviceIndex;       /// CUDA device index
    cudaDeviceProp              mProp;              /// CUDA device properties

    ThreadSafeQueue< CudaJobInfo* > mJobQueue;
    cudaStream_t                mStreamId[CUDA_STREAM_COUNT];          /// A list of available execution streams, used round-robin
    int                         mStreamIndex;       /// Index of the next stream to be used for submitting a job
    unique_ptr< std::thread >   mSubmissionThread;
    PlayoutJobInfo*             mInputHost;          /// Host-side job input buffer
    PlayoutJobResult*           mOutputHost;         /// Host-side job output buffer

    PlayoutJobInfo*             mInputDev;          /// Device-side job input buffer
    PlayoutJobResult*           mOutputDev;         /// Device-side job output buffer

    Semaphore mThreadStarted;
    Semaphore mThreadExited;

    CudaSubmissionThread( int deviceIndex )
    {
        mDeviceIndex = deviceIndex;

        mSubmissionThread = new std::thread( [] { this->SubmissionThread(); } );
        mThreadStarted.Wait();
    }

    ~CudaSubmissionThread()
    {
        mJobQueue.Push( NULL );
        mThreadExited.Wait();

        mSubmissionThread.join();

        if( mInputHost )
            cudaFreeHost( mInputHost );

        if( mOutputHost )
            cudaFreeHost( mOutputHost );

        if( mInputDev )
            cudaFree( mInputDev );

        if( mOutputDev )
            cudaFree( mOutputDev );

        for( size_t i = 0; i < mStreamId.size(); i++ )
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



    void SubmissionThread()
    {
        char threadName[80];
        sprintf( threadName, "CUDA %d", device->mDeviceIndex );
        PlatSetThreadName( threadName );

        this->InitCuda();

        mThreadStarted.Post();

        int warpSize = mProp.warpSize;

        for( ;; )
        {
            CudaJobInfo* job = mJobQueue.Pop();
            if( job == NULL )
                break;

            mStreamIndex = (mStreamIndex + 1) % mStreamId.size();
            job->mStream = mStreamId[mStreamIndex];

            extern void QueuePlayoutJobCuda( CudaJobInfo* job, int blockCount, int blockSize );

            QueuePlayoutJobCuda( job, blockCount, blockSize, exitFlag );
            job->mTickQueued = PlatGetClockTick();
        }

        mThreadDone.Post();
    }


};






#endif // CORVID_CUDA_HOST
#endif // ENABLE_CUDA
#endif // CORVID_GPU_H__

