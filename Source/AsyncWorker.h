

class AsyncWorker
{
protected:
    PlayoutJobQueue*    mIncoming;
    PlayoutJobQueue*    mOutgoing;

    unique_ptr< std::thread > mWorkerThread;
    volatile bool mExiting;
    Semaphore           mThreadExited;

    virtual void RunJob( PlayoutJobRef& job ) = 0;

    void RunJobs()
    {
        while( !mExiting )
        {
            PlayoutJobRef job;

            if( !mIncoming->TryPop( job, DEFAULT_ASYNC_POLL );
                continue;

            this->RunJob( job );

            mOutgoing->Push( job );
        }

        mThreadExited.Post();
    }

public:

    AsyncWorker( PlayoutJobQueue* workQueue, PlayoutJobQueue* doneQueue )
    {
        mWorkQueue = workQueue;
        mDoneQueue = doneQueue;
        mExiting  = false;

        mWorkerThread = new std::thread( [] { this->RunJobs(); } );
    }

    ~AsyncWorker()
    {
        mExiting = true;
        mThreadExited.Wait();
    }


};

class CpuWorker : public AsyncWorker
{
    override void RunJob( PlayoutJobRef& job )
    {
        GamePlayer player( &job->mOptions );

        job->mResult = player.Play( job->mPos, job->mNumPlays );
    }
}
