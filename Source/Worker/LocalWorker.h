// LocalWorker.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#pragma once


class LocalWorker : public AsyncWorker
{
    PlayoutJobQueue*        mJobQueue;
    PlayoutResultQueue*     mResultQueue;
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
            PlayoutJobRef job = mJobQueue->Pop();
            if( job == NULL )
                break;

            PlayoutResultRef result( new PlayoutResult() );
            result->mPathFromRoot = job.mPathFromRoot;

            extern ScoreCard PlayGamesX64( const PlayoutJob* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesSSE4( const PlayoutJob* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesAVX2( const PlayoutJob* job, PlayoutResult* result, int count );
            extern ScoreCard PlayGamesAVX512( const PlayoutJob* job, PlayoutResult* result, int count );

            int simdLevel   = ChooseSimdLevelForPlayout( job.mOptions, job.mNumGames );
            int simdCount   = (job.mNumGames + simdLevel - 1) / simdLevel;

            switch( simdLevel )
            {
            case 8:   result->mScores = PlayGamesAVX512( job, result, simdCount ); break;
            case 4:   result->mScores = PlayGamesAVX2( job, result, simdCount ); break;
            case 2:   result->mScores = PlayGamesSSE4( job, result, simdCount ); break;
            default:  result->mScores = PlayGamesX64( job, result, simdCount ); break;
            }

            mResultQueue->Push( result );
        }
    }

public:

    LocalWorker( const GlobalOptions* options, PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
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
