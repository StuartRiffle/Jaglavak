// TreeSearcher.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

struct TreeSearcher
{
    TreeNode*               mNodePool;
    size_t                  mNodePoolEntries;
    TreeLink                mMruListHead;
    TreeNode*               mSearchRoot;
    BranchInfo              mRootInfo;
    UciSearchConfig         mUciConfig;
    GlobalOptions*          mOptions;
    std::thread*            mSearchThread;
    Semaphore               mSearchThreadActive;
    Semaphore               mSearchThreadIdle;
    std::thread*            mResultThread;
    volatile bool           mShuttingDown;
    volatile bool           mSearchRunning;
    RandomGen               mRandom;
    PlayoutJobQueue         mJobQueue;
    PlayoutResultQueue      mResultQueue;
    std::vector< PlayoutResultRef > mResultBuffer;
    Mutex                   mResultBufferMutex;

    std::vector< std::shared_ptr< IAsyncWorker > > mAsyncWorkers;

    TreeSearcher( GlobalOptions* options, u64 randomSeed = 1 ) : 
        mOptions( options )
    {
        mSearchRoot    = NULL;
        mShuttingDown  = false;
        mSearchRunning = false;
        mRandom.SetSeed( randomSeed );

        mNodePoolEntries = mOptions->mMaxTreeNodes;
        mNodePool = new TreeNode[mNodePoolEntries];

        for( int i = 0; i < mNodePoolEntries; i++ )
        {
            mNodePool[i].mPrev = &mNodePool[i - 1];
            mNodePool[i].mNext = &mNodePool[i + 1];
            mNodePool[i].mCounter = -i;
        }

        mNodePool[0].mPrev = (TreeNode*) &mMruListHead;
        mMruListHead.mNext = &mNodePool[0];

        mNodePool[mNodePoolEntries - 1].mNext = (TreeNode*) &mMruListHead;
        mMruListHead.mPrev = &mNodePool[mNodePoolEntries - 1];
    }

    void Init()
    {
        for( int i = 0; i < mOptions->mNumCpuWorkers; i++ )
        {
            auto worker = new CpuWorker( mOptions, &mJobQueue, &mResultQueue );
            mAsyncWorkers.push_back( std::shared_ptr< IAsyncWorker >( worker ) );
        }

#if SUPPORT_CUDA
        if( mOptions->mAllowCuda )
        {
            for( int i = 0; i < GpuWorker::GetDeviceCount(); i++ )
            {
                auto worker = new GpuWorker( mOptions, &mJobQueue, &mResultQueue );
                worker->Initialize( i, mOptions->mCudaQueueDepth );

                mAsyncWorkers.push_back( std::shared_ptr< IAsyncWorker >( worker ) );
            }
        }
#endif

        mSearchThread  = new std::thread( [this] { this->SearchThread(); } );
        mResultThread  = new std::thread( [this] { this->ResultThread(); } );
    }

    ~TreeSearcher()
    {
        this->StopSearching();

        mShuttingDown = true;
        mSearchThreadActive.Post();

        mSearchThread->join();
        delete mSearchThread;

        mResultQueue.Push( NULL );
        mResultThread->join();
        delete mResultThread;

        delete mNodePool;
    }

    void MoveToFront( TreeNode* node )
    {
        TreeNode* oldFront = mMruListHead.mNext;

        assert( node->mNext->mPrev == node );
        assert( node->mPrev->mNext == node );
        assert( oldFront->mPrev == (TreeNode*) &mMruListHead );

        node->mNext->mPrev = node->mPrev;
        node->mPrev->mNext = node->mNext;

        node->mNext = mMruListHead.mNext;
        node->mNext->mPrev = node;

        node->mPrev = (TreeNode*) &mMruListHead;
        node->mPrev->mNext = node;

        assert( mMruListHead.mNext == node );

        static int touch = 0;
        node->mTouch = touch++;

                                    /*
        TreeNode* t = node->mPrev->mPrev->mPrev;
        for( int i = 0; i < 6; i++ )
        {
            printf( "%d ", t->mCounter );
            t = t->mNext;
        }
        printf( "\n" );
        */

        //this->DebugVerifyMruList();
    }

    TreeNode* AllocNode()
    {
        TreeNode* last = mMruListHead.mPrev;
        last->Clear();

        MoveToFront( last );

        TreeNode* first = mMruListHead.mNext;
        return first;
    }

    void SetPosition( const Position& startPos, const MoveList* moveList = NULL )
    {
        // TODO: recognize position and don't terf the whole tree

        Position pos = startPos;

        if( moveList )
            for( int i = 0; i < moveList->mCount; i++ )
                pos.Step( moveList->mMove[i] );

        if( mSearchRoot )
            mSearchRoot->mInfo = NULL;

        mSearchRoot = AllocNode();
        mSearchRoot->Init( pos );

        mSearchRoot->mInfo = &mRootInfo;
        mRootInfo.mNode = mSearchRoot;
    }

    void DebugVerifyMruList()
    {
        TreeNode* node = mMruListHead.mNext;
        int count = 0;

        while( node != (TreeNode*) &mMruListHead )
        {
            count++;
            node = node->mNext;
        }

        assert( count == mNodePoolEntries );
    }

    void Reset()
    {
        this->StopSearching();

        Position startPos;
        startPos.Reset();

        this->SetPosition( startPos );
    }

    double CalculateUct( TreeNode* node, int childIndex )
    {
        BranchInfo* nodeInfo    = node->mInfo;
        BranchInfo& childInfo   = node->mBranch[childIndex];
        const ScoreCard& scores = childInfo.mScores;

        int color = node->mColor;

        u64 draws           = scores.mPlays - (scores.mWins[0] + scores.mWins[1]);
        u64 childWins       = scores.mWins[color];
        u64 childPlays      = scores.mPlays;
        u64 nodePlays       = nodeInfo->mScores.mPlays;
        float exploringness = mOptions->mExplorationFactor * 1.0f / 100;

        if( nodePlays == 0 )
            nodePlays = 1;

        if( childPlays == 0 )
            childPlays = 1;

        double childWinRatio = (childWins + (draws * 0.5)) / childPlays;
        double uct = childWinRatio + exploringness * sqrt( log( nodePlays * 1.0 ) * 2.0 / childPlays );

        return uct;
    }

    int SelectNextBranch( TreeNode* node )
    {
        int numBranches = (int) node->mBranch.size();
        assert( numBranches > 0 );

        // Choose an untried branch at random

        int idx = (int) mRandom.GetRange( numBranches );

        for( int i = 0; i < numBranches; i++ )
        {
            if( !node->mBranch[idx].mNode )
                return idx;

            if( ++idx == numBranches )
                idx = 0;
        }

        // This node is fully expanded, so choose 
        // the move with highest UCT

        double highestUct = 0;
        int highestIdx = 0;

        for( int i = 0; i < numBranches; i++ )
        {
            double uct = CalculateUct( node, i );
            if( uct > highestUct )
            {
                highestUct = uct;
                highestIdx = i;
            }
        }

        return highestIdx;
    }

    ScoreCard ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node )
    {
        //node->SanityCheck();

        MoveToFront( node );

        if( node->mGameOver )
        {
            node->mInfo->mScores += node->mGameResult;
            return( node->mGameResult );
        }

        int chosenBranchIdx = SelectNextBranch( node );
        BranchInfo* chosenBranch = &node->mBranch[chosenBranchIdx];

        pathFromRoot.Append( chosenBranch->mMove );

        /*
        DEBUG_LOG( "ExpandAtLeaf %s choosing %d/%d (%s)\n",
            pathFromRoot.mCount? SerializeMoveList( pathFromRoot ).c_str() : "(root)",
            chosenBranchIdx, node->mBranch.size(), SerializeMoveSpec( chosenBranch->mMove ).c_str() );
            */

        if( !chosenBranch->mNode )
        {
            // This is a leaf, so create a new node 

            TreeNode* newNode = AllocNode();
            assert( newNode != node );

            MoveToFront( node );

            Position newPos = node->mPos;
            newPos.Step( chosenBranch->mMove );

            //DEBUG_LOG("%s STEPPING %s\n", SerializePosition( newPos ).c_str(), SerializeMoveSpec( chosenBranch->mMove ).c_str() );

            chosenBranch->mNode = NULL;

            newNode->Init( newPos, chosenBranch ); 

            // newNode is different, node is the same value

            chosenBranch->mNode = newNode;

            if( newNode->mGameOver )
            {
                newNode->mInfo->mScores += newNode->mGameResult;
                return( newNode->mGameResult );
            }

            //DEBUG_LOG( "Running playouts: %s [%d]\n", SerializeMoveList( pathFromRoot ).c_str(), newPos.mWhiteToMove );

            ScoreCard scores;
            PlayoutJob job;

            job.mOptions        = *mOptions;
            job.mRandomSeed     = mRandom.GetNext();
            job.mPosition       = newPos;
            job.mNumGames       = mOptions->mNumInitialPlays;
            job.mPathFromRoot   = pathFromRoot;
            
            if( mOptions->mNumInitialPlays > 0 )
            {
                // Do the initial playouts

                PlayoutResult jobResult = RunPlayoutJobCpu( job );
                scores += jobResult.mScores;
            }
            else
            {
                // (or just pretend we did)
                
                scores.mPlays = 1;
            }

            if( mOptions->mNumAsyncPlays > 0 )
            {
                // Queue up any async playouts

                PlayoutJobRef asyncJob( new PlayoutJob() );

                *asyncJob = job;
                asyncJob->mNumGames = mOptions->mNumAsyncPlays;

                mJobQueue.Push( asyncJob );
            }

            newNode->mInfo->mScores += scores;

            return scores;
        }

        ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, chosenBranch->mNode );

        // Accumulate the scores on our way back down the tree

        chosenBranch->mScores += branchScores;

        // Mark each node LRU on the way

        MoveToFront( node );

        return branchScores;
    }


    void ExpandAtLeaf()
    {
        PROFILER_SCOPE( "TreeSearcher::ExpandAtLeaf" );

        MoveList pathFromRoot;
        ScoreCard rootScores = this->ExpandAtLeaf( pathFromRoot, mSearchRoot );

        //MoveToFront( mSearchRoot );

        mSearchRoot->mInfo->mScores += rootScores;

        int chosenBranch = mSearchRoot->FindMoveIndex( pathFromRoot.mMove[0] );
        assert( chosenBranch >= 0 );
    }


    void ProcessResult( TreeNode* node, const PlayoutResultRef& result, int depth = 0 )
    {
        if( node == NULL )
            return;

        if( depth >= result->mPathFromRoot.mCount )
            return;

        MoveSpec move = result->mPathFromRoot.mMove[depth];

        int childIdx = node->FindMoveIndex( move );
        if( childIdx < 0 )
            return;

        TreeNode* child = node->mBranch[childIdx].mNode;
        if( child == NULL )
            return;

        ProcessResult( child, result, depth + 1 );

        node->mBranch[childIdx].mScores += result->mScores;
    }

    void DumpStats( TreeNode* node )
    {
        u64 bestDenom = 0;
        int bestDenomIdx = 0;

        float bestRatio = 0;
        int bestRatioIdx = 0;

        for( int i = 0; i < (int) node->mBranch.size(); i++ )
        {
            if( node->mBranch[i].mScores.mPlays > bestDenom )
            {
                bestDenom = node->mBranch[i].mScores.mPlays;
                bestDenomIdx = i;
            }

            if( node->mBranch[i].mScores.mPlays > 0 )
            {
                float ratio = node->mBranch[i].mScores.mWins[node->mColor] * 1.0f / node->mBranch[i].mScores.mPlays;

                if( ratio > bestRatio )
                {
                    bestRatio = ratio;
                    bestRatioIdx = i;
                }
            }
        }

        printf( "Queue length %d\n", mJobQueue.GetCount() );
        for( int i = 0; i < (int) node->mBranch.size(); i++ )
        {
            std::string moveText = SerializeMoveSpec( node->mBranch[i].mMove );
            printf( "%s%s  %2d) %5s %.15f %12ld/%-12ld\n", 
                (i == bestRatioIdx)? ">" : " ", 
                (i == bestDenomIdx)? "***" : "   ", 
                i,
                moveText.c_str(), 
                this->CalculateUct( node, i ), 
                node->mBranch[i].mScores.mWins[node->mColor], node->mBranch[i].mScores.mPlays );
        }
    }

    void UpdateAsyncWorkers()
    {
        PROFILER_SCOPE( "TreeSearcher::UpdateAsyncWorkers" );

        for( auto& worker : mAsyncWorkers )
            worker->Update();
    }

    void ProcessAsyncResults()
    {
        PROFILER_SCOPE( "TreeSearcher::ProcessAsyncResults" );


        MUTEX_SCOPE( mResultBufferMutex );

        for( auto& result : mResultBuffer )
            this->ProcessResult( mSearchRoot, result );

        mResultBuffer.clear();
    }

    void ResultThread()
    {
        PlatBoostThreadPriority();
        PlatSetThreadName( "Result" );

        for( ;; )
        {
            PlayoutResultRef result = mResultQueue.Pop();
            if( result == NULL )
                break;

            MUTEX_SCOPE( mResultBufferMutex );
            mResultBuffer.push_back( result );
        }
    }



    void SearchThread()
    {
        // Make sure we don't get interrupted by worker threads

        PlatBoostThreadPriority();
        PlatSetThreadName( "Search" );

        Timer timer;

        for( ;; )
        {
            // Wait until we're needed

            mSearchThreadIdle.Post();
            mSearchThreadActive.Wait();

            // Run until we're not

            if( mShuttingDown )
                break;

            int counter = 0;

            while( mSearchRunning )
            {
                PROFILER_SCOPE( "TreeSearcher::SearchThread" );

                if( (counter % 10000) == 0 )
                {
                    u64 total = 0;
                    for( int i = 0; i < (int) mSearchRoot->mBranch.size(); i++ )
                        total += mSearchRoot->mBranch[i].mScores.mPlays;

                    float secs = timer.GetElapsedMs()/ 1000.0f;
                    float rate = (secs > 0)? (total / secs) : 0;

                    printf( "\n%d iters, %d games/sec\n", counter, (int) rate );
                    this->DumpStats( mSearchRoot );
                }
                counter++;

                this->UpdateAsyncWorkers();
                this->ProcessAsyncResults();

                //if( mJobQueue.GetCount() >= mOptions->mMaxPendingJobs )
                //    PlatYield();

                if( mJobQueue.GetCount() < mOptions->mMaxPendingJobs )
                    this->ExpandAtLeaf();

                PlatYield();
            }
        }
    }

    void StartSearching( const UciSearchConfig& config )
    {
        this->StopSearching();

        mUciConfig = config;

        mSearchRunning  = true;
        mSearchThreadActive.Post();
    }

    void StopSearching()
    {
        if( mSearchRunning )
        {
            mSearchRunning = false;
            mSearchThreadIdle.Wait();
        }
    }
};


