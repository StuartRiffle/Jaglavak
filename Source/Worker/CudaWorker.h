// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class CudaWorker : public AsyncWorker
{
    typedef std::map< cudaStream_t, std::list< CudaLaunchSlot* > > LaunchSlotsByStream;

    int                             mDeviceIndex;      
    std::unique_ptr< CudaLauncher > mLaunchThread;
    std::unique_ptr< std::thread >  mJobThread;
    std::vector< CudaLaunchSlot >   mSlotInfo;      
    const GlobalOptions*            mOptions;

    Mutex                           mSlotMutex;
    std::vector< CudaLaunchSlot* >  mFreeSlots;
    LaunchSlotsByStream             mActiveSlotsByStream;


    bool mInitialized;
    BatchQueue*    mJobQueue;
    BatchQueue* mResultQueue;

public:    
    CudaWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions = options;
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;
        mInitialized = false;
    }

    ~CudaWorker()
    {
        if( mInitialized )
            this->Shutdown();
    }

    static int GetDeviceCount()
    {
        int count = 0;
        cudaGetDeviceCount( &count );

        return( count );
    }

    void Initialize( int deviceIndex, int jobSlots )
    {
        mDeviceIndex  = deviceIndex;
        cudaSetDevice( deviceIndex );

        mLaunchThread = std::unique_ptr< CudaLauncher >( new CudaLauncher( mOptions, deviceIndex ) );
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
            std::vector< PlayoutBatch > jobs = mJobQueue.PopBulk( mOptions->mCudaJobBatch, true );
            if( jobs.empty() )
                break;

            bool freeSlotAvail = false;

            MUTEX_SCOPE( mSlotMutex );

            if( mFreeSlots.empty() )
            {
                PlatSleep( mOptions)
            }

            mSlotMutex.Enter();
            bool empty = mFreeSlots.empty();
            mSlotMutex.Leave();

            if( empty )

            if( empty )
            {
                PlatSleep( 100 );
                continue;
            }

            PlayoutBatch job = mJobQueue->Pop();
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

                cudaEventElapsedTime( &result->mGpuTime, slot->mStartEvent, slot->mReadyEvent );
                result->mCpuLatency = (tickReturned - slot->mTickQueued) * 1000.0f / PlatGetClockFrequency();  

                mResultQueue->Push( result );

                mFreeSlots.push_back( slot );
            }
        }
    }
};
  
