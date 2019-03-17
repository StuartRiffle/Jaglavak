// LocalWorker.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_LOCAL_WORKER_H__
#define CORVID_LOCAL_WORKER_H__


class IAsyncWorker
{
public:
    virtual void Update() {}
};



class LocalWorker : public IAsyncWorker
{
    PlayoutJobQueue*    mJobQueue;
    PlayoutResultQueue* mResultQueue;
    std::thread*        mJobThread;

    void JobThread()
    {
        for( ;; )
        {
            PlayoutJobRef job = mJobQueue->Pop();
            if( job == NULL )
                break;

            ScoreCard scores = PlayGamesCpu( job->mOptions, job->mPosition, job->mNumGames );

            PlayoutResultRef result( new PlayoutResult() );

            result->mScores = scores;
            result->mPathFromRoot = job->mPathFromRoot;

            mResultQueue->Push( result );
        }
    }

public:

    LocalWorker( PlayoutJobQueue* jobQueue, PlayoutResultQueue* doneQueue )
    {
        mJobQueue = jobQueue;
        mResultQueue = doneQueue;

        mJobThread = new std::thread( [&] { this->JobThread(); } );
    }

    ~LocalWorker()
    {
        // Owner will kill the job thread by feeding it a NULL

        mJobThread->join();
        delete mJobThread;
    }
};

#endif // CORVID_LOCAL_WORKER_H__

