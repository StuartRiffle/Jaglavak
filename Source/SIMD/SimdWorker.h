// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "SimdPlayer.h"

class SimdWorker : public AsyncWorker
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
            if( !mWorkQueue->Pop( batch ) )
                break;

            int count = (int) batch->mPosition.size();
            int paddedCount = count + SIMD_WIDEST * 2;

            batch->mResults.resize( paddedCount ); // TODO: align this

            PlayGamesSimd( 
                mOptions, 
                &batch->mParams, 
                batch->mPosition.data(), 
                batch->mResults.data(),
                count );

            batch->mResults.resize( count );
            mDoneQueue->Push( batch );
        }
    }

public:

    SimdWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions    = options;
        mWorkQueue  = jobQueue;
        mDoneQueue  = resultQueue;
        mWorkThread = unique_ptr< thread >( new thread( [this] { this->WorkThread(); } ) );
    }

    ~SimdWorker()
    {
        mWorkThread->join();
    }
};
