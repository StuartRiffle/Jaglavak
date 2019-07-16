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
    float _GpuTime;
};

typedef shared_ptr< LaunchInfo > LaunchInfoRef;

class CudaWorker : public AsyncWorker
{
    enum
    {
        CUDA_NUM_STREAMS = 16
    };

    const GlobalSettings*   _Settings;
    BatchQueue*             _BatchQueue;

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
    CudaWorker( const GlobalSettings* settings, BatchQueue* queue );
    ~CudaWorker();

    static int CudaWorker::GetDeviceCount()
    {
        int count = 0;
        auto res = cudaGetDeviceCount( &count );
        return( count );
    }

    const cudaDeviceProp& GetDeviceProperties() { return _Prop; }
    bool Initialize( int deviceIndex );
    void Shutdown();

private:
    cudaEvent_t AllocEvent();
    void FreeEvent( cudaEvent_t event );

    void ___CUDA_LAUNCH_THREAD___();
    virtual void Update() override;
};
  
