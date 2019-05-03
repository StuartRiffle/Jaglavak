// LocalWorker.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_CPU_WORKER_H__
#define CORVID_CPU_WORKER_H__

class IAsyncWorker
{
public:
    virtual void Update() {}
};

class CpuWorker : public IAsyncWorker
{
    PlayoutJobQueue*        mJobQueue;
    PlayoutResultQueue*     mResultQueue;
    std::thread*            mJobThread;
    const GlobalOptions*    mOptions;

    void JobThread()
    {
        PlatSetThreadName( "CpuWorker" );

        for( ;; )
        {
            PlayoutJobRef job = mJobQueue->Pop();
            if( job == NULL )
                break;

            PlayoutResultRef result( new PlayoutResult() );
            *result = RunPlayoutJobCpu( *job );

            mResultQueue->Push( result );
        }

        printf( "WTF\n" );
    }

public:

    CpuWorker( const GlobalOptions* options, PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
    {
        mOptions = options;
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;

        mJobThread = new std::thread( [this] { this->JobThread(); } );
    }

    ~CpuWorker()
    {
        // Owner will kill the job thread by feeding it a NULL

        mJobThread->join();
        delete mJobThread;
    }
};

#endif // CORVID_CPU_WORKER_H__

