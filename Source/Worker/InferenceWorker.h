// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class JobBase
{
    virtual void Run() {}

};

struct InferenceJob
{
    string mGraphName;
    string mCheckpoint;
    MoveList mPathFromRoot;
    vector< float > mInputs;
    vector< float > mOutputs;
};


class InferenceWorker : public AsyncWorker
{
    const GlobalOptions*    mOptions;
    InferenceJobQueue*      mWorkQueue;
    InferenceJobQueue*      mDoneQueue;
    PTR< thread >           mJobThread;

    void JobThread()
    {
        for( ;; )
        {
            JobRef job;
            if( !mWorkQueue->PopBlocking( job ) )
                break;

            TensorFlowSession* session = this->GetSession( job->mGraphName, job->mCheckpoint );
            assert( session );

            if( session )
            {
                session->Run()

                mDoneQueue->PushBlocking( job );
            }
        }
    }

public:

    InferenceWorker( const GlobalOptions* options, InferenceJobQueue* workQueue, InferenceJobQueue* doneQueue )
    {
        mOptions = options;
        mWorkQueue = workQueue;
        mDoneQueue = doneQueue;

        mJobThread = PTR< thread* >( new thread( [this] { this->JobThread(); } ) );
    }

    ~InferenceWorker()
    {
        mJobThread->join();
    }
};
