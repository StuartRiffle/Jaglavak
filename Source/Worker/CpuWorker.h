// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class CpuWorker : public AsyncWorker
{
    const GlobalSettings*   _Settings   = NULL;
    Metrics*                _Metrics    = NULL;
    BatchQueue*             _BatchQueue = NULL;
    volatile bool           _TimeToExit = false;
    
    list< unique_ptr< thread > > _WorkThreads;

public:

    CpuWorker( const GlobalSettings* settings, Metrics* metrics, BatchQueue* batchQueue )
    {
        _Settings = settings;
        _Metrics = metrics;
        _BatchQueue = batchQueue;
    }

    ~CpuWorker()
    {
        _TimeToExit = true;
        _BatchQueue->NotifyAllWaiters();

        for( auto& thread : _WorkThreads )
            thread->join();
    }

    virtual bool Initialize();

private:

    void PrintCpuInfo();
    void ___CPU_WORK_THREAD___( int idx );
};
