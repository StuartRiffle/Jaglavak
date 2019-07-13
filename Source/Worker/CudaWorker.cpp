// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"

#include "CudaSupport.h"
#include "CudaPlayer.h"
#include "CudaWorker.h"

CudaWorker::CudaWorker( const GlobalSettings* settings, BatchQueue* batchQueue )
{
    _Settings = settings;
    _BatchQueue = batchQueue;
    _ShuttingDown = false;
}

CudaWorker::~CudaWorker()
{
    this->Shutdown();
}

bool CudaWorker::Initialize( int deviceIndex  )
{
    _DeviceIndex = deviceIndex;
    _StreamIndex = 0;

    cudaError_t status = cudaGetDeviceProperties( &_Prop, _DeviceIndex );
    if( status != cudaSuccess )
        return false;

    cudaSetDevice( _DeviceIndex );
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    int totalCores = _Prop.multiProcessorCount * GetCudaCoresPerSM( _Prop.major, _Prop.minor );
    u64 mb = _Prop.totalGlobalMem / (1024 * 1024);
    u64 mhz = _Prop.clockRate / 1000;

    cout << "CUDA " << _DeviceIndex << ": " << _Prop.name << endl;
    cout << "  Compute  " << _Prop.major << "." << _Prop.minor << endl;
    cout << "  Clock    " << mhz << " MHz" << endl;
    cout << "  Memory   " << mb << " MB" << endl;
    cout << "  Cores    " << totalCores << endl << endl;

    int megs = _Settings->Get( "CUDA.HeapMegs" );

    _Heap.Init( megs * 1024 * 1024 );
    _LaunchThread = unique_ptr< thread >( new thread( [this] { this->LaunchThread(); } ) );

    return true;
}

void CudaWorker::Shutdown()
{
    _ShuttingDown = true;
    _BatchQueue->NotifyAllWaiters();
    _LaunchThread->join();

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        cudaStreamDestroy( _StreamId[i] );

    for( auto& event : _EventCache )
        cudaEventDestroy( event );
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
    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
        CUDA_REQUIRE(( cudaStreamCreateWithFlags( _StreamId + i, cudaStreamNonBlocking ) ));

    for( ;; )
    {
        int batchesToGrab = MAX( 1, _Settings->Get( "CUDA.BatchSize" ) / _Settings->Get( "Search.BatchSize" ) );

        // The actual batches can be different sizes, but close enough. Pop
        // them all at once to minimize queue traffic.

        vector< BatchRef > newBatches = _BatchQueue->PopMulti( batchesToGrab );
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

        int streamIndex = _StreamIndex++;
        _StreamIndex %= CUDA_NUM_STREAMS;
        cudaStream_t stream = _StreamId[streamIndex];

        int totalWidth = total * _Settings->Get( "Search.NumPlayouts" );
        int blockSize  = _Prop.warpSize;
        int blockCount = (totalWidth + blockSize - 1) / blockSize;

        launch->_StartTimer = this->AllocEvent();
        launch->_StopTimer  = this->AllocEvent(); 
        launch->_ReadyEvent = this->AllocEvent();

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

        launch->_TickSubmitted = CpuInfo::GetClockTick();
        _InFlightByStream[streamIndex].push_back( launch );
    }
}

void CudaWorker::Update() 
{
    // This is called from the main thread to process completed batches
    
    unique_lock< mutex > lock( _Mutex );

    for( int i = 0; i < CUDA_NUM_STREAMS; i++ )
    {
        auto& inFlight = _InFlightByStream[i];
        while( !inFlight.empty() )
        {
            LaunchInfoRef launch = inFlight.front();
            if( cudaEventQuery( launch->_ReadyEvent ) != cudaSuccess )
                break;

            inFlight.pop_front();
            launch->_TickReturned = CpuInfo::GetClockTick();

            int offset = 0;
            for( auto& batch : launch->_Batches )
            {
                ScoreCard* results = (ScoreCard*) &launch->_Outputs[offset];

                batch->_GameResults.assign( results, results + batch->GetCount() );
                offset += batch->GetCount();

                batch->_Done = true;
            }

            _Heap.Free( launch->_Params );
            _Heap.Free( launch->_Inputs );
            _Heap.Free( launch->_Outputs );

            this->FreeEvent( launch->_StartTimer );
            this->FreeEvent( launch->_StopTimer );
            this->FreeEvent( launch->_ReadyEvent );
        }
    }
}
