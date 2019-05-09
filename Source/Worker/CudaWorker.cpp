// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "CudaSupport.h"
#include "CudaWorker.h"


CudaWorker::CudaWorker( const GlobalOptions* options, BatchQueue* workQueue, BatchQueue* doneQueue )
{
    mOptions = options;
    mWorkQueue = workQueue;
    mDoneQueue = doneQueue;
    mShuttingDown = false;
}

CudaWorker::~CudaWorker()
{
    this->Shutdown();
}

// static
int CudaWorker::GetDeviceCount()
{
    int count = 0;
    cudaGetDeviceCount( &count );

    return( count );
}

void CudaWorker::Initialize( int deviceIndex, int jobSlots )
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

    mLaunchThread = unique_ptr< thread >( new thread( [this] { this->LaunchThread(); } ) );
}

void CudaWorker::Shutdown()
{
    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        cudaStreamDestroy( mStreamId[i] );

    for( auto& slot : mSlotInfo )
        cudaEventDestroy( slot.mReadyEvent );

    mShuttingDown = true;
    mVar.notify_all();        
    mLaunchThread->join();
}

CudaLaunchSlot* CudaWorker::ClaimFreeSlot()
{
    unique_lock< mutex > lock( mMutex );

    while( mFreeSlots.empty() )
    {
        mVar.wait( lock );
        if( mShuttingDown )
            return( NULL );
    }

    CudaLaunchSlot* slot = mFreeSlots.back();
    mFreeSlots.pop_back();

    return slot;
}

void CudaWorker::LaunchThread()
{
    CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));

    for( ;; )
    {
        CudaLaunchSlot* slot = this->ClaimFreeSlot();
        if( !slot )
            break;

        BatchRef batch;
        if( !mWorkQueue->PopBlocking( batch ) )
            break;

        slot->mBatch = batch;

        int streamIndex = mStreamIndex++;
        mStreamIndex %= CUDA_NUM_STREAMS;

        cudaStream_t stream = mStreamId[streamIndex];

        slot->mParams.CopyToDeviceAsync( stream );
        slot->mInputs.CopyToDeviceAsync( stream );
        slot->mOutputs.ClearOnDeviceAsync( stream );

        int totalWidth = slot->mCount * batch->mParams.mNumGamesEach;
        int blockCount = (totalWidth + mProp.warpSize - 1) / mProp.warpSize;

        PlayGamesCudaAsync( 
            slot->mParams.mDevice, 
            slot->mInputs.mDevice, 
            slot->mOutputs.mDevice, 
            slot->mCount,
            blockCount, 
            mProp.warpSize, 
            stream );

        slot->mOutputs.CopyToHostAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( slot->mReadyEvent, stream ) ));

        mActiveSlotsByStream[streamIndex].push_back( slot );
    }
}

void CudaWorker::Update() 
{
    // This is called from the main thread

    unique_lock< mutex > lock( mMutex );
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
            for( int i = 0; i < batch->GetCount(); i++ )
                batch->mResults[i] = slot->mOutputs[i];

            mFreeSlots.push_back( slot );
            completedBatches.push_back( batch );
        }
    }

    lock.unlock();
    mVar.notify_all();

    mDoneQueue->PushMulti( completedBatches );
}
