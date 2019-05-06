// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once



struct CudaLauncher
{


    void LaunchThread()
    {
        this->InitCuda();
        mLaunchThreadRunning.Post();

        for( ;; )
        {
            CudaLaunchSlot* slot;
            if( !mLaunchQueue.Pop( slot ) )
                break;


        }

        this->ShutdownCuda();
    }

public:
    CudaLauncher( const GlobalOptions* options, int deviceIndex )
    {
        mOptions = options;
        mDeviceIndex = deviceIndex;
    }

    ~CudaLauncher()
    {
        mLaunchQueue.Terminate();

        if( mLaunchThread )
            mLaunchThread->join();
    }

    void Init()
    {
        mLaunchThread = std::unique_ptr< std::thread >( new std::thread( [this] { this->LaunchThread(); } ) );
        mLaunchThreadRunning.Wait();
    }

    void Launch( CudaLaunchSlot* slot )
    {
        mLaunchQueue.Push( slot );
    }
};
