// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Player/CpuPlayer.h"

class CpuWorker : public AsyncWorker
{
    const GlobalOptions*    _Options;
    BatchQueue*             _BatchQueue;
    volatile bool           _TimeToExit;
    list< unique_ptr< thread > > _WorkThreads;

public:

    SimdWorker( const GlobalOptions* options, BatchQueue* batchQueue )
    {
        _Options = options;
        _BatchQueue = batchQueue;
        _TimeToExit = false;
    }

    virtual bool Initialize()
    {
        cout << "CPU: " << CpuInfo::GetCpuName() << endl;
        cout << "  Cores    " << CpuInfo::DetectCpuCores() << endl;
        cout << "  Sockets  " << "2" << endl; // FIXME
        cout << "  SIMD     " << CpuInfo::DetectSimdLevel() << "x (" << simdName << ")" << endl << endl;

        for( int i = 0; i < _Options->_CpuWorkThreads; i++ )
            _WorkThreads.emplace_back( new thread( [this] { this->WorkThread(); } );

        return (_WorkThreads.size() > 0);
    }

    ~SimdWorker()
    {
        _TimeToExit = true;
        _Queue->NotifyAll();

        for( auto& thread : _WorkThreads )
            thread->join();
    }

private:
    void WorkThread()
    {
        while( !_TimeToExit )
        {
            BatchRef batch;
            if( !_Queue->Pop( batch ) )
                break;

            int count = (int) batch->_Position.size();
            batch->_GameResults.resize( count + SIMD_MAX );

            PlayGamesSimd( 
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
