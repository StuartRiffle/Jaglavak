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

    mLaunchThread = unique_ptr< thread >( new thread( [this] { this->LaunchThread(); } ) );
}

void CudaWorker::Shutdown()
{
    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        cudaStreamDestroy( mStreamId[i] );

    mShuttingDown = true;
    mVar.notify_all();        
    mLaunchThread->join();
}

cudaEvent_t CudaWorker::AllocEvent()
{
    cudaEvent_t result = NULL;

    if(mEventCache.empty())
    {
        auto status = cudaEventCreate( &result );
        assert( status == cudaSuccess );
    }
    else
    {
        cudaEvent_t result = mEventCache.back();
        mEventCache.pop_back();
    }

    return result;
}

void CudaWorker::FreeEvent( cudaEvent_t event )
{
    mEventCache.push_back( event );
}

void CudaWorker::LaunchThread()
{
    CUDA_REQUIRE(( cudaSetDevice( mDeviceIndex ) ));

    for( ;; )
    {
        vector< BatchRef > batches = mWorkQueue->PopMultiBlocking( mOptions->mCudaBatchesPerLaunch );
        if( batches.empty() )
            break;

        if( mShuttingDown )
            break;

        unique_lock< mutex > lock( mMutex );

        LaunchInfoRef launch( new LaunchInfo() );
        launch->mBatches = batches;

        // Combine the batches into one big buffer

        int total = 0;
        for( auto& batch : batches )
            total += batch->GetCount();

        mHeap.Alloc( total, &launch->mParams );
        mHeap.Alloc( total, &launch->mInputs );
        mHeap.Alloc( total, &launch->mOutputs );

        int offset = 0;
        for( auto& batch : batches )
        {
            int count = batch->GetCount();
            for( int i = 0; i < count; i++ )
            {
                launch->mInputs[offset + i] = batch->mPosition[i];
                launch->mParams[offset + i] = batch->mParams;
            }
            offset += count;
        }

        launch->mStartTimer = this->AllocEvent();
        launch->mStopTimer  = this->AllocEvent(); 
        launch->mReadyEvent = this->AllocEvent();

        int streamIndex = mStreamIndex++;
        mStreamIndex %= CUDA_NUM_STREAMS;
        cudaStream_t stream = mStreamId[streamIndex];

        int totalWidth = total * mOptions->mNumAsyncPlayouts;
        int blockSize  = mProp.warpSize;
        int blockCount = (totalWidth + blockSize - 1) / blockSize;

        launch->mParams.CopyUpToDeviceAsync( stream );
        launch->mInputs.CopyUpToDeviceAsync( stream );
        launch->mOutputs.ClearOnDeviceAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( launch->mStartTimer, stream ) ));

        PlayGamesCudaAsync( 
            launch->mParams.mDevice, 
            launch->mInputs.mDevice, 
            launch->mOutputs.mDevice, 
            total,
            blockCount, 
            blockSize, 
            stream );

        CUDA_REQUIRE(( cudaEventRecord( launch->mStopTimer, stream ) ));
        launch->mOutputs.CopyDownToHostAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( launch->mReadyEvent, stream ) ));

        mInFlightByStream[streamIndex].push_back( launch );
    }
}

void CudaWorker::Update() 
{
    unique_lock< mutex > lock( mMutex );

    // This is called from the main thread to gather completed batches

    vector< BatchRef > completed;

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
    {
        auto& running = mInFlightByStream[i];
        while( !running.empty() )
        {
            LaunchInfoRef& launch = running.front();
            if( cudaEventQuery( launch->mReadyEvent ) != cudaSuccess )
                break;

            running.pop_front();

            int offset = 0;
            for( auto& batch : launch->mBatches )
            {
                ScoreCard* results = (ScoreCard*) &launch->mOutputs[offset];

                batch->mResults.assign( results, results + batch->GetCount() );
                offset += batch->GetCount();

                completed.push_back( batch );
            }

            mHeap.Free( launch->mParams );
            mHeap.Free( launch->mInputs );
            mHeap.Free( launch->mOutputs );

            this->FreeEvent( launch->mStartTimer );
            this->FreeEvent( launch->mStopTimer );
            this->FreeEvent( launch->mReadyEvent );
        }
    }

    mDoneQueue->Push( completed );
}
