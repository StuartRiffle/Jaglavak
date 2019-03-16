

class IAsyncWorker
{
public:
    virtual void Update() {}
};


class CpuAsyncWorker : public IAsyncWorker
{
    PlayoutJobQueue*    mIncoming;
    PlayoutResultQueue* mOutgoing;

    unique_ptr< std::thread > mWorkerThread;

    void JobThread()
    {
        for( ;; )
        {
            PlayoutJobRef job = mIncoming->Pop();
            if( job == NULL )
                break;

            GamePlayer player( &job->mOptions );
            ScoreCard scores = player.Play( job->mPos, job->mNumPlays );

            PlayoutJobResultRef result = new PlayoutJobResult();
            result->mScores = scores;
            result->mPathFromRoot = job->mPathFromRoot;

            mOutgoing->Push( result );
        }
    }

public:

    CpuAsyncWorker( PlayoutJobQueue* workQueue, PlayoutResultQueue* doneQueue )
    {
        mIncoming = workQueue;
        mOutgoing = doneQueue;

        mWorkerThread = new std::thread( [] { this->JobThread(); } );
    }

    ~CpuAsyncWorker()
    {
        mWorkerThread.join();
    }
};

