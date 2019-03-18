// TreeSearcher.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

struct TreeSearcher
{
    TreeNode*               mNodePool;
    size_t                  mNodePoolEntries;
    TreeLink                mMruListHead;
    TreeNode*               mSearchRoot;
    UciSearchConfig         mUciConfig;
    GlobalOptions*          mOptions;
    std::thread*            mSearchThread;
    Semaphore               mSearchThreadActive;
    Semaphore               mSearchThreadIdle;
    volatile bool           mShuttingDown;
    volatile bool           mSearchRunning;
    RandomGen               mRandom;
    PlayoutJobQueue         mJobQueue;
    PlayoutResultQueue      mResultQueue;

    vector< shared_ptr< IAsyncWorker > > mAsyncWorkers;

    TreeSearcher( GlobalOptions* options, u64 randomSeed ) : mOptions( options )
    {
        mShuttingDown  = false;
        mSearchRunning = false;
        mRandom.SetSeed( randomSeed );
        mSearchThread  = new std::thread( [&] { this->SearchThread(); } );

        mNodePoolEntries = mOptions->mMaxTreeNodes;
        mNodePool = new TreeNode[mNodePoolEntries];

        for( int i = 0; i < mNodePoolEntries; i++ )
        {
            mNodePool[i].mPrev = &mNodePool[i - 1];
            mNodePool[i].mNext = &mNodePool[i + 1];
        }

        mNodePool[0].mPrev = (TreeNode*) &mMruListHead;
        mMruListHead.mNext = &mNodePool[0];

        mNodePool[mNodePoolEntries - 1].mNext = (TreeNode*) &mMruListHead;
        mMruListHead.mPrev = &mNodePool[mNodePoolEntries - 1];

        this->Reset();
    }

    ~TreeSearcher()
    {
        this->StopSearching();

        mShuttingDown = true;
        mSearchThreadActive.Post();

        mSearchThread->join();
        delete mSearchThread;

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

        node->mNext = oldFront;
        node->mPrev = (TreeNode*) &mMruListHead;

        node->mNext->mPrev = node;
        node->mPrev->mNext = node;

        assert( mMruListHead.mNext == node );

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

        mSearchRoot = AllocNode();
        mSearchRoot->Init( pos );
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

    float CalculateUct( TreeNode* node, int childIndex )
    {
        BranchInfo* nodeInfo    = node->mInfo;
        BranchInfo& childInfo   = node->mBranch[childIndex];

        u64 childWins       = childInfo.mScores.mWins;
        u64 childPlays      = childInfo.mScores.mPlays;
        u64 nodePlays       = nodeInfo->mScores.mPlays;
        float exploringness = mOptions->mExplorationFactor;

        float childWinRatio = childWins * 1.0f / childPlays;
        float uct = childWinRatio + exploringness * sqrtf( logf( nodePlays ) / childPlays );

        return uct;
    }

    int SelectNextBranch( TreeNode* node )
    {
        assert( node->mBranch.size() > 0 );

        // Just take the move with the highest UCT

        float highestUct = node->mBranch[0].mUct;
        int highestIdx = 0;

        for( int i = 1; i < (int) node->mBranch.size(); i++ )
        {
            if( node->mBranch[i].mUct > highestUct )
            {
                highestUct = node->mBranch[i].mUct;
                highestIdx = i;
            }
        }

        return highestIdx;
    }

    ScoreCard ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node )
    {
        // Mark each node LRU as we walk up the tree

        MoveToFront( node );

        int nextBranchIdx = SelectNextBranch( node );
        BranchInfo& nextBranch = node->mBranch[nextBranchIdx];

        pathFromRoot.Append( nextBranch.mMove );

        if( !nextBranch.mNode )
        {
            // This is a leaf, so create a new node 

            TreeNode* newNode = AllocNode();
            newNode->Init( node->mPos, &nextBranch ); 

            nextBranch.mNode = newNode;

            node->mNumChildren++;
            assert( node->mNumChildren <= (int) node->mBranch.size() );
           
            // Expand one of its moves at random

            int newBranchIdx = mRandom.GetRange( newNode->mBranch.size() );

            BranchInfo& newBranch = newNode->mBranch[newBranchIdx];
            pathFromRoot.Append( newBranch.mMove );

            Position newPos = node->mPos;
            newPos.Step( newBranch.mMove );

            ScoreCard scores;
            scores.Clear();

            PlayoutJob job;

            job.mOptions        = mOptions;
            job.mRandomSeed     = mRandom.GetNext();
            job.mPosition       = newPos;
            job.mNumGames       = mOptions->mNumInitialPlays;
            job.mPathFromRoot   = pathFromRoot;
            
            if( mOptions->mNumInitialPlays > 0 )
            {
                // Do the initial playouts

                JobResult jobResult = RunPlayoutJobCpu( job );
                scores += jobResult.mScores;
            }
            else
            {
                // ...or just pretend we did
                
                scores.mPlays = 1;
                scores.mDraws = 1;
            }

            if( mOptions->mNumAsyncPlays > 0 )
            {
                // Queue up any async playouts

                PlayoutJobRef asyncJob( new PlayoutJob() );

                *asyncJob = job;
                asyncJob->mNumGames = mOptions->mNumAsyncPlays;

                // This will BLOCK when the job queue fills up

                mJobQueue.Push( job );
            }

            newBranch.mScores += scores;
            newBranch.mUct = CalculateUct( node, newBranchIdx );

            scores.FlipColor();
            return scores;
        }

        ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, nextBranch.mNode );

        // Accumulate the scores on our way back down the tree

        nextBranch.mScores += branchScores;
        nextBranch.mUct = CalculateUct( node, nextBranchIdx );

        branchScores.FlipColor();
        return branchScores;
    }

    void ProcessResult( TreeNode* node, const PlayoutResultRef& result, int depth = 0 )
    {
        if( node == NULL )
            return;

        MoveSpec move = result->mPathFromRoot.mMove[depth];

        int childIdx = node->FindMoveIndex( move );
        if( childIdx < 0 )
            return;

        TreeNode* child = node->mBranch[childIdx].mNode;
        if( child == NULL )
            return;

        ProcessResult( child, result, depth + 1 );

        ScoreCard scores = result->mScores;

        bool otherColor = (result->mPathFromRoot.mCount & 1) != 0;
        if( otherColor )
            scores.FlipColor();

        node->mBranch[childIdx].mScores += scores;
        node->mBranch[childIdx].mUct = CalculateUct( node, childIdx );
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

        vector< PlayoutResultRef > results = mResultQueue.PopAll();

        for( const auto& result : results )
            this->ProcessResult( mSearchRoot, result );
    }

    void ExpandAtLeaf()
    {
        PROFILER_SCOPE( "TreeSearcher::ExpandAtLeaf" );

        MoveList pathFromRoot;
        this->ExpandAtLeaf( pathFromRoot, mSearchRoot );
    }

    void SearchThread()
    {
        for( ;; )
        {
            // Wait until we're needed

            mSearchThreadIdle.Post();
            mSearchThreadActive.Wait();

            // Run until we're not

            if( mShuttingDown )
                break;

            while( mSearchRunning )
            {
                PROFILER_SCOPE( "TreeSearcher::SearchThread" );

                //this->UpdateAsyncWorkers();
                //this->ProcessAsyncResults();

                this->ExpandAtLeaf();
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


