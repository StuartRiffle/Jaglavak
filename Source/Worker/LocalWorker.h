// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class LocalWorker : public AsyncWorker
{
    BatchQueue*        mJobQueue;
    BatchQueue*     mResultQueue;
    std::thread*            mJobThread;
    const GlobalOptions*    mOptions;

    int ChooseSimdLevelForPlayout( const GlobalOptions& options, int count )
    {
        int simdLevel = 1;

        if( (count > 1) && (options.mDetectedSimdLevel >= 2) )
            simdLevel = 2;

        if( (count > 2) && (options.mDetectedSimdLevel >= 4) )
            simdLevel = 4;

        if( (count > 4) && (options.mDetectedSimdLevel >= 8) )
            simdLevel = 8;

        if( !options.mAllowSimd )
            simdLevel = 1;

        if( options.mForceSimdLevel )
            simdLevel = options.mForceSimdLevel;

        return simdLevel;
    }

    void JobThread()
    {
        for( ;; )
        {
            PlayoutBatchRef job = mJobQueue->Pop();
            if( job == NULL )
                break;

            int simdLevel   = ChooseSimdLevelForPlayout( job.mOptions, job.mNumGames );
            int simdCount   = (job.mNumGames + simdLevel - 1) / simdLevel;

            extern ScoreCard PlayGamesAVX512( const PlayoutBatch* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesAVX2(   const PlayoutBatch* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesSSE4(   const PlayoutBatch* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesX64(    const PlayoutBatch* job, PlayoutResult* result, int count );

            PlayoutResultRef result( new PlayoutResult() );

            switch( simdLevel )
            {
            case 8:   PlayGamesAVX512( job, result, simdCount ); break;
            case 4:   PlayGamesAVX2(   job, result, simdCount ); break;
            case 2:   PlayGamesSSE4(   job, result, simdCount ); break;
            default:  PlayGamesX64(    job, result, simdCount ); break;
            }

            mResultQueue->Push( result );
        }
    }

public:

    LocalWorker( const GlobalOptions* options, BatchQueue* jobQueue, BatchQueue* resultQueue )
    {
        mOptions = options;
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;

        mJobThread = new std::thread( [this] { this->JobThread(); } );
    }

    ~LocalWorker()
    {
        mJobThread->join();
    }
};
