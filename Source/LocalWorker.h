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

            PlayoutResultRef result( new PlayoutResult() );
            *result = RunPlayoutJobCpu( *job );

            mResultQueue->Push( result );
        }
    }

public:

    LocalWorker( PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
    {
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;

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

