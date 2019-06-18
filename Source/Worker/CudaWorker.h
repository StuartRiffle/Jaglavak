// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "CudaSupport.h"

struct LaunchInfo
{
    vector< BatchRef >          _Batches;
    CudaBuffer< PlayoutParams > _Params;
    CudaBuffer< Position >      _Inputs;
    CudaBuffer< ScoreCard >     _Outputs;

    u64 _TickSubmitted;
    u64 _TickReturned;

    cudaEvent_t _StartTimer;
    cudaEvent_t _StopTimer; 
    cudaEvent_t _ReadyEvent; 
};

typedef shared_ptr< LaunchInfo > LaunchInfoRef;

class CudaWorker : public AsyncWorker
{
    enum
    {
        CUDA_NUM_STREAMS = 16
    };

    const GlobalOptions*    _Options;
    BatchQueue*             _Queue;

    int                     _DeviceIndex;      
    cudaDeviceProp          _Prop;
    CudaAllocator           _Heap;

    unique_ptr< thread >    _LaunchThread;
    bool                    _ShuttingDown;

    mutex                   _Mutex;
    condition_variable      _Var;

    int                     _StreamIndex;
    cudaStream_t            _StreamId[CUDA_NUM_STREAMS];
    list< LaunchInfoRef >   _InFlightByStream[CUDA_NUM_STREAMS];
    vector< cudaEvent_t >   _EventCache;

public:    
    CudaWorker( const GlobalOptions* options, BatchQueue* queue );
    ~CudaWorker();

    static int GetDeviceCount();

    const cudaDeviceProp& GetDeviceProperties() { return _Prop; }
    bool Initialize( int deviceIndex );
    void Shutdown();

private:
    cudaEvent_t AllocEvent();
    void FreeEvent( cudaEvent_t event );

    void LaunchThread();
    virtual void Update() override;
};
  
