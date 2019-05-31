// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "SimdPlayer.h"

class SimdWorker : public AsyncWorker
{
    const GlobalOptions*    _Options;
    BatchQueue*             _WorkQueue;
    BatchQueue*             _DoneQueue;
    unique_ptr< thread >    _WorkThread;

    void WorkThread()
    {
        for( ;; )
        {
            BatchRef batch;
            if( !_WorkQueue->Pop( batch ) )
                break;

            int count = (int) batch->_Position.size();
            int paddedCount = count + SIMD_WIDEST * 2;

            batch->_GameResults.resize( paddedCount ); // TODO: align this

            PlayGamesSimd( 
                _Options, 
                &batch->_Params, 
                batch->_Position.data(), 
                batch->_GameResults.data(),
                count );

            batch->_GameResults.resize( count );
            _DoneQueue->Push( batch );
        }
    }

public:

    SimdWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        _Options    = options;
        _WorkQueue  = jobQueue;
        _DoneQueue  = resultQueue;
        _WorkThread = unique_ptr< thread >( new thread( [this] { this->WorkThread(); } ) );
    }

    ~SimdWorker()
    {
        _WorkThread->join();
    }
};
