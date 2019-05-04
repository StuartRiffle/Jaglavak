// LocalWorker.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef JAGLAVAK_CPU_WORKER_H__
#define JAGLAVAK_CPU_WORKER_H__

class LocalWorker : public IAsyncWorker
{
    PlayoutJobQueue*        mJobQueue;
    PlayoutResultQueue*     mResultQueue;
    std::thread*            mJobThread;
    const GlobalOptions*    mOptions;

    void JobThread()
    {
        PlatSetThreadName( "LocalWorker" );

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

    LocalWorker( const GlobalOptions* options, PlayoutJobQueue* jobQueue, PlayoutResultQueue* resultQueue )
    {
        mOptions = options;
        mJobQueue = jobQueue;
        mResultQueue = resultQueue;

        mJobThread = new std::thread( [this] { this->JobThread(); } );
    }

    ~LocalWorker()
    {
        // Owner will kill the job thread by feeding it a NULL

        mJobThread->join();
        delete mJobThread;
    }
};

#endif // JAGLAVAK_CPU_WORKER_H__

