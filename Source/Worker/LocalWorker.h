// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

extern void PlayGamesAVX512( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesAVX2(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesSSE4(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
extern void PlayGamesX64(    const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count );

class LocalWorker : public AsyncWorker
{
    const GlobalOptions*    mOptions;
    BatchQueue*             mWorkQueue;
    BatchQueue*             mDoneQueue;
    std::unique_ptr< std::thread* > mWorkThread;

    int ChooseSimdLevelForPlayout( int count )
    {
        int simdLevel = 1;

        if( (count > 1) && (mOptions->mDetectedSimdLevel >= 2) )
            simdLevel = 2;

        if( (count > 2) && (mOptions->mDetectedSimdLevel >= 4) )
            simdLevel = 4;

        if( (count > 4) && (mOptions->mDetectedSimdLevel >= 8) )
            simdLevel = 8;

        if( !mOptions->mAllowSimd )
            simdLevel = 1;

        if( mOptions->mForceSimdLevel )
            simdLevel = mOptions->mForceSimdLevel;

        return simdLevel;
    }

    void JobThread()
    {
        for( ;; )
        {
            BatchRef batch;
            if( !mWorkQueue->PopBlocking( batch ) )
                break;

            int simdLevel   = ChooseSimdLevelForPlayout( batch->mCount );
            int simdCount   = (batch->mCount + simdLevel - 1) / simdLevel;

            switch( simdLevel )
            {
            case 8:   PlayGamesAVX512( &batch->mParams, batch->mPos, batch->mResults, simdCount ); break;
            case 4:   PlayGamesAVX2(   &batch->mParams, batch->mPos, batch->mResults, simdCount ); break;
            case 2:   PlayGamesSSE4(   &batch->mParams, batch->mPos, batch->mResults, simdCount ); break;
            default:  PlayGamesX64(    &batch->mParams, batch->mPos, batch->mResults, simdCount ); break;
            }

            mDoneQueue->PushBlocking( batch );
        }
    }

public:

    LocalWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions = options;
        mWorkQueue = jobQueue;
        mDoneQueue = resultQueue;

        mWorkThread = std::unique_ptr< std::thread* >( new std::thread( [this] { this->JobThread(); } ) );
    }

    ~LocalWorker()
    {
        mWorkThread->join();
    }
};
