// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

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
    auto res = cudaGetDeviceCount( &count );

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

        CUDA_REQUIRE(( cudaEventCreate( &slot.mReadyEvent ) ));
        mFreeSlots.push_back( &slot );

        slot.mNumLaunches = 0;
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

        int count = batch->GetCount();
        int totalWidth = count * batch->mParams.mNumGamesEach;

        slot->mBatch = batch;
        slot->mParams[0] = batch->mParams;
        slot->mParams[0].mNumGamesEach = 1;

        assert( slot->mInputs.mBufferSize  >= (count * sizeof( Position )) );
        assert( slot->mOutputs.mBufferSize >= (count * sizeof( ScoreCard )) );

        for( int i = 0; i < count; i++ )
            slot->mInputs[i] = batch->mPosition[i];

        int streamIndex = mStreamIndex++;
        mStreamIndex %= CUDA_NUM_STREAMS;
        cudaStream_t stream = mStreamId[streamIndex];

        slot->mParams.CopyToDeviceAsync( stream );
        slot->mInputs.CopyToDeviceAsync( stream );
        slot->mOutputs.ClearOnDeviceAsync( stream );

        int blockSize = mProp.warpSize;
        int blockCount = (totalWidth + blockSize - 1) / blockSize;

        PlayGamesCudaAsync( 
            slot->mParams.mDevice, 
            slot->mInputs.mDevice, 
            slot->mOutputs.mDevice, 
            batch->GetCount(),
            blockCount, 
            blockSize, 
            stream );

        slot->mOutputs.CopyToHostAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( slot->mReadyEvent, stream ) ));

        mActiveSlotsByStream[streamIndex].push_back( slot );
        slot->mNumLaunches++;
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
            assert( batch->mResults.size() == 0 );
            batch->mResults.reserve( batch->GetCount() );

            for( int i = 0; i < batch->GetCount(); i++ )
            {
                ScoreCard& scores = slot->mOutputs[i];
                batch->mResults.push_back( scores );
            }

            mFreeSlots.push_back( slot );
            completedBatches.push_back( batch );
        }
    }

    lock.unlock();
    mVar.notify_all();

    mDoneQueue->Push( completedBatches );
}
