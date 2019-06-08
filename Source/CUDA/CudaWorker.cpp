// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

#include "CudaSupport.h"
#include "CudaWorker.h"


CudaWorker::CudaWorker( const GlobalOptions* options, BatchQueue* workQueue, BatchQueue* doneQueue )
{
    _Options = options;
    _WorkQueue = workQueue;
    _DoneQueue = doneQueue;
    _ShuttingDown = false;
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

void CudaWorker::Initialize( int deviceIndex  )
{
    _DeviceIndex = deviceIndex;
    _StreamIndex = 0;

    CUDA_REQUIRE(( cudaSetDevice( _DeviceIndex ) ));
    CUDA_REQUIRE(( cudaGetDeviceProperties( &_Prop, _DeviceIndex ) ));
    CUDA_REQUIRE(( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) ));

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        CUDA_REQUIRE(( cudaStreamCreateWithFlags( _StreamId + i, cudaStreamNonBlocking ) ));

    _Heap.Init( (u64) _Options->_CudaHeapMegs * 1024 * 1024 );
    _LaunchThread = unique_ptr< thread >( new thread( [this] { this->LaunchThread(); } ) );
}

void CudaWorker::Shutdown()
{
    _ShuttingDown = true;
    _WorkQueue->NotifyAllWaiters();
    _DoneQueue->NotifyAllWaiters();
                     
    _LaunchThread->join();

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        cudaStreamDestroy( _StreamId[i] );
}

cudaEvent_t CudaWorker::AllocEvent()
{
    cudaEvent_t result = NULL;

    if( _EventCache.empty() )
    {
        auto status = cudaEventCreate( &result );
        assert( status == cudaSuccess );
    }
    else
    {
        result = _EventCache.back();
        _EventCache.pop_back();
    }

    return result;
}

void CudaWorker::FreeEvent( cudaEvent_t event )
{
    _EventCache.push_back( event );
}

void CudaWorker::LaunchThread()
{
    CUDA_REQUIRE(( cudaSetDevice( _DeviceIndex ) ));

    for( ;; )
    {
        vector< BatchRef > newBatches = _WorkQueue->PopMulti( _Options->_CudaBatchesPerLaunch );
        if( _ShuttingDown )
            break;
        if( newBatches.empty() )
            break;

        LaunchInfoRef launch( new LaunchInfo() );
        launch->_Batches = newBatches;

        unique_lock< mutex > lock( _Mutex );

        // Combine the batches into one big buffer

        int total = 0;
        for( auto& batch : launch->_Batches )
            total += batch->GetCount();

        _Heap.Alloc( total, &launch->_Params );
        _Heap.Alloc( total, &launch->_Inputs );
        _Heap.Alloc( total, &launch->_Outputs );
                                     
        int offset = 0;
        for( auto& batch : launch->_Batches )
        {
            int count = batch->GetCount();
            for( int i = 0; i < count; i++ )
            {
                launch->_Inputs[offset + i] = batch->_Position[i];
                launch->_Params[offset + i] = batch->_Params;
            }
            offset += count;
        }

        launch->_StartTimer = this->AllocEvent();
        launch->_StopTimer  = this->AllocEvent(); 
        launch->_ReadyEvent = this->AllocEvent();

        int streamIndex = _StreamIndex++;
        _StreamIndex %= CUDA_NUM_STREAMS;
        cudaStream_t stream = _StreamId[streamIndex];

        int totalWidth = total * _Options->_NumAsyncPlayouts;
        int blockSize  = _Prop.warpSize;
        int blockCount = (totalWidth + blockSize - 1) / blockSize;

        launch->_Params.CopyUpToDeviceAsync( stream );
        launch->_Inputs.CopyUpToDeviceAsync( stream );
        launch->_Outputs.ClearOnDeviceAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( launch->_StartTimer, stream ) ));

        PlayGamesCudaAsync( 
            launch->_Params._Device, 
            launch->_Inputs._Device, 
            launch->_Outputs._Device, 
            total,
            blockCount, 
            blockSize, 
            stream );

        CUDA_REQUIRE(( cudaEventRecord( launch->_StopTimer, stream ) ));
        launch->_Outputs.CopyDownToHostAsync( stream );
        CUDA_REQUIRE(( cudaEventRecord( launch->_ReadyEvent, stream ) ));

        _InFlightByStream[streamIndex].push_back( launch );
    }
}

void CudaWorker::Update() 
{
    unique_lock< mutex > lock( _Mutex );

    // This is called from the main thread to gather completed batches

    vector< BatchRef > completed;

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
    {
        auto& inFlight = _InFlightByStream[i];
        while( !inFlight.empty() )
        {
            LaunchInfoRef launch = inFlight.front();
            if( cudaEventQuery( launch->_ReadyEvent ) != cudaSuccess )
                break;

            inFlight.pop_front();

            int offset = 0;
            for( auto& batch : launch->_Batches )
            {
                ScoreCard* results = (ScoreCard*) &launch->_Outputs[offset];

                batch->_GameResults.assign( results, results + batch->GetCount() );
                offset += batch->GetCount();

                completed.push_back( batch );
            }

            _Heap.Free( launch->_Params );
            _Heap.Free( launch->_Inputs );
            _Heap.Free( launch->_Outputs );

            this->FreeEvent( launch->_StartTimer );
            this->FreeEvent( launch->_StopTimer );
            this->FreeEvent( launch->_ReadyEvent );
        }
    }

    _DoneQueue->Push( completed );
}
