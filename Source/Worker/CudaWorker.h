// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class CudaWorker : public AsyncWorker
{
    enum
    {
        CUDA_NUM_STREAMS = 16
    };

    const GlobalOptions*        mOptions;
    BatchQueue*                 mWorkQueue;
    BatchQueue*                 mDoneQueue;

    int                         mDeviceIndex;      
    cudaDeviceProp              mProp;
    unique_ptr< thread >        mLaunchThread;
    bool                        mShuttingDown;

    mutex                       mMutex;
    condition_variable          mVar;
    vector< CudaLaunchSlot >    mSlotInfo;
    vector< CudaLaunchSlot* >   mFreeSlots;

    int                         mStreamIndex;
    cudaStream_t                mStreamId[CUDA_NUM_STREAMS];
    list< CudaLaunchSlot* >     mActiveSlotsByStream[CUDA_NUM_STREAMS];

public:    
    CudaWorker( const GlobalOptions* options, BatchQueue* workQueue, BatchQueue* doneQueue );
    ~CudaWorker();

    static int GetDeviceCount();
    const cudaDeviceProp& GetDeviceProperties() { return mProp; }
    void Initialize( int deviceIndex, int jobSlots );
    void Shutdown();

private:
    CudaLaunchSlot* ClaimFreeSlot();
    void LaunchThread();
    virtual void Update() override;
};
  
