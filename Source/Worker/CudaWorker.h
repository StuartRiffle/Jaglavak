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

extern void PlayGamesCudaAsync( CudaLaunchSlot* slot, int blockCount, int blockSize, cudaStream_t stream );


class CudaWorker : public AsyncWorker
{
    enum
    {
        CUDA_NUM_STREAMS = 16,
    };

    const GlobalOptions*        mOptions;
    BatchQueue*                 mWorkQueue;
    BatchQueue*                 mDoneQueue;

    int                         mDeviceIndex;      
    cudaDeviceProp              mProp;
    PTR< thread >               mLaunchThread;
    bool                        mShuttingDown;

    mutex                       mMutex;
    condition_variable          mVar;
    vector< CudaLaunchSlot >    mSlotInfo;
    vector< CudaLaunchSlot* >   mFreeSlots;

    int                         mStreamIndex;
    cudaStream_t                mStreamId[CUDA_NUM_STREAMS];
    list< CudaLaunchSlot* >     mActiveSlotsByStream[CUDA_NUM_STREAMS];

    public:    
    CudaWorker( const GlobalOptions* options, BatchQueue* workQueue, BatchQueue* doneQueue )
    {
        mOptions = options;
        mWorkQueue = workQueue;
        mDoneQueue = doneQueue;
        mShuttingDown = false;
    }

    ~CudaWorker()
    {
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
        mDeviceIndex = deviceIndex;
        mStreamIndex = 0;

        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaGetDeviceProperties( &mProp, mDeviceIndex ) ));
        CUDA_REQUIRE(( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) ));

        for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
            CUDA_REQUIRE(( cudaStreamCreateWithFlags( mStreamId + i, cudaStreamNonBlocking ) ));

        mSlotInfo.resize( jobSlots );
        for( int i = 0; i < jobSlots; i++ )
        {
            CudaLaunchSlot& slot = mSlotInfo[i];

            slot.mParams.Init( 1 );
            slot.mInputs.Init( PLAYOUT_BATCH_MAX );
            slot.mOutputs.Init( PLAYOUT_BATCH_MAX );
            slot.mCount = 0;

            CUDA_REQUIRE(( cudaEventCreate( &slot.mReadyEvent ) ));
            mFreeSlots.push_back( &slot );
        }

        mLaunchThread = PTR< thread >( new thread( [this] { this->LaunchThread(); } ) );
    }

    void Shutdown()
    {
        for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
            cudaStreamDestroy( mStreamId[i] );

        for( auto& slot : mSlotInfo )
            cudaEventDestroy( slot.mReadyEvent );

        mShuttingDown = true;
        mVar.notify_all();        
        mLaunchThread->join();
    }

private:

    CudaLaunchSlot* ClaimFreeSlot()
    {
        lock_guard< mutex > lock( mMutex );

        while( mFreeSlots.empty() )
        {
            mVar.wait();
            if( mShuttingDown )
                return( NULL );
        }

        CudaLaunchSlot* slot = mFreeSlots.back();
        mFreeSlots.pop_back();

        return slot;
    }

    void LaunchThread()
    {
        CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));

        for( ;; )
        {
            BatchRef batch;
            if( !mWorkQueue->Pop( batch ) )
                break;

            CudaLaunchSlot* slot = this->ClaimFreeSlot();
            if( !slot )
                break;

            slot->mBatch = batch;

            int streamIndex = mStreamIndex++;
            mStreamIndex %= CUDA_NUM_STREAMS;

            cudaStream_t stream = mStreamId[streamIndex];

            slot->mParams.CopyToDeviceAsync( stream );
            slot->mInputs.CopyToDeviceAsync( stream );
            slot->mOutputs.ClearOnDeviceAsync( stream );

            int totalWidth = batch->mCount * batch->mParams->mNumGamesEach;
            int blockCount = (totalWidth + mProp.warpSize - 1) / mProp.warpSize;

            PlayGamesCudaAsync( slot, blockCount, mProp.warpSize, stream );

            slot->mOutputs.CopyToHostAsync( stream );
            CUDA_REQUIRE(( cudaEventRecord( slot->mReadyEvent, stream ) ));

            mActiveSlotsByStream[streamIndex].push_back( slot );
        }
    }

    virtual void Update() override
    {
        // This is called from the main thread

        lock_guard< mutex > lock( mMutex );
        vector< BatchRef > completedBatches;

        for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        {
            auto& activeList = mActiveSlotsByStream[i];
            while( !activeList.empty() )
            {
                CudaLaunchSlot* slot = activeList.front();
                if( cudaEventQuery( slot->mReadyEvent ) != cudaSuccess )
                    break;

                activeList.pop_front();

                BatchRef batch = slot->mBatch;
                for( int i = 0; i < batch->mCount; i++ )
                    batch->mResults[i] = slot->mOutputBuffer[i];

                mFreeSlots.push_back( slot );
                completedBatches.push_back( batch );
            }
        }

        lock.unlock();
        mVar.notify_all();

        mDoneQueue->PushMulti( completedBatches );
    }
};
  
