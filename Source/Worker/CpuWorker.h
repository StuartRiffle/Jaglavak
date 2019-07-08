// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Player/CpuPlayer.h"
#include "boost/algorithm/string.hpp"

class CpuWorker : public AsyncWorker
{
    const GlobalSettings*   _Settings   = NULL;
    BatchQueue*             _BatchQueue = NULL;
    volatile bool           _TimeToExit = false;
    
    list< unique_ptr< thread > > _WorkThreads;

public:

    CpuWorker( const GlobalSettings* settings, BatchQueue* batchQueue )
    {
        _Settings = settings;
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
    void WorkThread();
};
