// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

extern void PlayGamesAVX512( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesAVX2(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesSSE4(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesX64(    const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count );

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
            if( !mWorkQueue->PopBlocking( batch ) )
                break;

            size_t count = batch->GetCount();
            assert( count == batch->mPathFromRoot.size() );
            assert( count <= PLAYOUT_BATCH_MAX );

            int simdLevel = mOptions->mDetectedSimdLevel;
            if( !mOptions->mEnableSimd )
                simdLevel = 1;
            if( mOptions->mForceSimdLevel )
                simdLevel = mOptions->mForceSimdLevel;

            int simdCount = (count + simdLevel - 1) / simdLevel;
            batch->mResults.resize( count );

            switch( simdLevel )
            {
            case 8:   PlayGamesAVX512( &batch->mParams, batch->mPosition.data(), batch->mResults.data(), simdCount ); break;
            case 4:   PlayGamesAVX2(   &batch->mParams, batch->mPosition.data(), batch->mResults.data(), simdCount ); break;
            case 2:   PlayGamesSSE4(   &batch->mParams, batch->mPosition.data(), batch->mResults.data(), simdCount ); break;
            default:  PlayGamesX64(    &batch->mParams, batch->mPosition.data(), batch->mResults.data(), count ); break;
            }

            mDoneQueue->Push( batch );
        }
    }

public:

    SimdWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions = options;
        mWorkQueue = jobQueue;
        mDoneQueue = resultQueue;

        mWorkThread = unique_ptr< thread >( new thread( [this] { this->WorkThread(); } ) );
    }

    ~SimdWorker()
    {
        mWorkThread->join();
    }
};