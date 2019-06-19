// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Player/CpuPlayer.h"

class CpuWorker : public AsyncWorker
{
    const GlobalOptions*    _Options    = NULL;
    BatchQueue*             _BatchQueue = NULL;
    volatile bool           _TimeToExit = false;
    
    list< unique_ptr< thread > > _WorkThreads;

public:

    CpuWorker( const GlobalOptions* options, BatchQueue* batchQueue )
    {
        _Options = options;
        _BatchQueue = batchQueue;
    }

    ~CpuWorker()
    {
        _TimeToExit = true;
        _BatchQueue->NotifyAllWaiters();

        for( auto& thread : _WorkThreads )
            thread->join();
    }

    virtual bool Initialize()
    {
        string cpuName   = CpuInfo::GetCpuName();
        int    cores     = CpuInfo::DetectCpuCores();
        int    simdLevel = CpuInfo::DetectSimdLevel();
        string simdDesc  = CpuInfo::GetSimdDesc( simdLevel );

        cout << 
            "CPU: " << cpuName << endl <<
            "  Cores    " << cores << endl <<
            "  SIMD     " << simdLevel << "x (" << simdDesc << ")" << endl << endl;

        _TimeToExit = false;
        for( int i = 0; i < _Options->_CpuWorkThreads; i++ )
            _WorkThreads.emplace_back( new thread( [this]() { WorkThread(); } ) );

        return (_WorkThreads.size() > 0);
    }

private:
    void WorkThread()
    {
        while( !_TimeToExit )
        {
            BatchRef batch;
            if( !_BatchQueue->Pop( batch ) )
                break;

            int count = (int) batch->_Position.size();
            batch->_GameResults.resize( count + SIMD_WIDEST );

            PlayGamesCpu( 
                _Options, 
                &batch->_Params, 
                batch->_Position.data(), 
                batch->_GameResults.data(),
                count );

            batch->_GameResults.resize( count );
            batch->_Done = true;
        }
    }
};
