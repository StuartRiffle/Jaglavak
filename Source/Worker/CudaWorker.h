// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct LaunchInfo
{
    BatchRef mBatch;

    CudaBuffer< PlayoutRequest > mInputs;
    CudaBuffer< PlayoutResult >  mOutputs;

    cudaEvent_t mStartTimer;
    cudaEvent_t mStopTimer; 
    cudaEvent_t mReadyEvent; 
};

typedef shared_ptr< LaunchInfo > LaunchInfoRef;

class CudaWorker : public AsyncWorker
{
    enum
    {
        CUDA_MAX_STREAMS = 16
    };

    const GlobalOptions*    mOptions;
    BatchQueue*             mWorkQueue;
    BatchQueue*             mDoneQueue;

    int                     mDeviceIndex;      
    cudaDeviceProp          mProp;
    CudaAllocator           mHeap;

    unique_ptr< thread >    mLaunchThread;
    bool                    mShuttingDown;

    mutex                   mMutex;
    condition_variable      mVar;

    int                     mStreamIndex;
    cudaStream_t            mStreamId[CUDA_MAX_STREAMS];
    list< LaunchInfoRef >   mInFlightByStream[CUDA_MAX_STREAMS];

public:    
    CudaWorker( const GlobalOptions* options, BatchQueue* workQueue, BatchQueue* doneQueue );
    ~CudaWorker();

    static int GetDeviceCount();
    const cudaDeviceProp& GetDeviceProperties() { return mProp; }
    void Initialize( int deviceIndex, int jobSlots );
    void Shutdown();

private:
    void LaunchThread();
    virtual void Update() override;
};
  
