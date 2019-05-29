// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "SimdSwitch.h"

class CpuWorker : public AsyncWorker
{
    const GlobalOptions*    mOptions;
    BatchQueue*             mWorkQueue;
    BatchQueue*             mDoneQueue;
    unique_ptr< thread >    mWorkThread;

    void WorkThread()
    {
        for( ;; )
        {
            BatchRef batch;
            if( !mWorkQueue->PopBlocking( batch ) )
                break;

            size_t count = batch->mPosition.size();
            assert( count <= PLAYOUT_BATCH_MAX );

            ScoreCard ALIGN_SIMD scores[PLAYOUT_BATCH_MAX];
            memset( scores, 0, sizeof( ScoreCard ) * count );

            PlayGamesSimd( mOptions, &batch->mParams, batch->mPosition.data(), scores, count );

            assert( batch->mResults.size() == 0 );
            batch->mResults.assign( scores, scores + count );

            mDoneQueue->Push( batch );
        }
    }

public:

    CpuWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions    = options;
        mWorkQueue  = jobQueue;
        mDoneQueue  = resultQueue;
        mWorkThread = unique_ptr< thread >( new thread( [this] { this->WorkThread(); } ) );
    }

    ~CpuWorker()
    {
        mWorkThread->join();
    }
};
