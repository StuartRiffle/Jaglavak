struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    float       mUCT;

    BranchInfo()
    {
        mNode   = NULL;
        mUCT    = 1.0f;
    }
};

struct TreeNode
{
    Position            mPos;
    BranchInfo*         mInfo;
    int                 mNumChildren;
    vector<BranchInfo>  mBranch;

    TreeNode() : mInfo( NULL ), mChildren( 0 ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL )
    {
        mPos = pos;
        mInfo = info;
        mChildren = 0;

        MoveList moveList;
        moveList.FindMoves( pos );

        mBranch.clear();
        mBranch.resize( moveList.mCount );

        for( int i = 0; i < moveList.mCount; i++ )
            mBranch[i].mMove = moveList.mMove[i];
    }

    void Clear()
    {
        assert( this->IsLeaf() );

        if( mInfo )
        {
            assert( mInfo->mNode == this );
            mInfo->mNode = NULL;
        }

        mInfo = NULL;
        mNumChildren = 0;
        mBranch.clear();
    }

    int FindMoveIndex( const MoveSpec& move )
    {
        for( int i = 0; i < (int) mBranch.size(); i++ )
            if( mBranch[i].mMove == move )
                return( i );

        return( -1 );
    }

    bool IsLeaf() const
    {
        return (mNumChildren == 0);
    }

    bool IsFull() const
    {
        return (mNumChildren == (int) mBranch.size());
    }
};



struct TreeSearcher
{
    list< TreeNode >    mNodes;
    TreeNode*           mRoot;
    UciSearchConfig     mUciConfig;
    SearchOptions       mOptions;

    TreeSearcher( SearchOptions* options )
    {
        mOptions = options;
        this->Reset();
    }

    ~TreeSearcher()
    {
        DeleteAllNodes();
    }

    void MoveToFront( list< TreeNode >::iterator iter )
    {
        mNodes.splice( mNodes.begin(), mNodes, iter );
    }

    TreeNode* AllocNode()
    {
        int limit = mOptions.mNodeLimit;

        while( mNodes.size() > limit )
            mNodes.pop_back();

        if( mNodes.size() == limit )
        {
            // Recycle the last node

            MoveToFront( --mNodes.end() );
            mNodes.front().Clear();
        }
        else
        {
            // Create a new one

            mNodes.emplace_front();
        }

        return &mNodes.front();
    }

    void DeleteAllNodes()
    {
        while( !mNodes.empty() )
            mNodes.pop_back();

        mRoot = NULL;
    }

    void SetPosition( const Position& startPos, const MoveList* moveList = NULL )
    {
        // TODO: recognize position and don't terf the whole tree

        Position pos = startPos;

        if( moveList )
            for( int i = 0; i < moveList->mCount; i++ )
                pos.ApplyMove( moveList->mMove[i] );

        mRoot = AllocNode();
        mRoot->Init( pos );
    }

    void Reset()
    {
        this->DeleteAllNodes();

        Position startPos;
        startPos.Reset();

        this->SetPosition( startPos );
    }

    float CalculateUct( BranchInfo& branch, int childIndex )
    {
        BranchInfo& child = branch.mBranch[childIndex];

        float childWinRatio = child.mScores.mWins * 1.0f / child.mScores.mPlays;
        float uct = childWinRatio + mOptions.mExplorationFactor * sqrtf( logf( branch.mScores.mPlays ) / child.mScores.mPlays );

        return uct;
    }

    int SelectNextBranch( TreeNode* node )
    {
        // Just take the move with the highest UCT

        float highestUct = node->mBranch[0].mUct;
        int highestIdx = 0;

        for( int i = 1; i < movesToPeek; i++ )
        {
            if( node->mBranch[i].mUct > highestUct )
            {
                highestUct = eval[i];
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
        BranchInfo& nextBranch = mNode->mBranch[nextBranchIdx];

        pathFromRoot->Add( info.mMove );

        if( !nextBranch.mNode )
        {
            // This is a leaf, so create a new node 

            TreeNode* newNode = AllocNode();
            newNode->Init( parent->mPos, &nextBranch ); 

            nextBranch.mNode = newNode;
            node->mChildren++;
           
            // Expand one of its moves at random

            int newBranchIdx = mOptions.mRandom.Get( newNode->mBranch.size() );

            BranchInfo& newBranch = newNode->mBranch[newBranchIdx];
            pathFromRoot.Add( newBranch.mMove );

            // Queue up the bulk async playouts

            if( mOptions.mAllowAsyncPlayouts )
            {
                Position newPos = newBranch.mPos;
                newPos.ApplyMove( newBranch.mMove );

                PlayoutJobInfo job;

                job.mPosition       = newPos;
                job.mOptions        = mOptions.mPlayout;
                job.mNumGames       = mOptions.mNumAsyncPlays;
                job.mPathFromRoot   = pathFromRoot;

                mPlayoutJobQueue.Push( job );
            }

            // Do the initial playouts

            ScoreCard scores = PlayGamesCpu( mOptions.mPlayoutOptions, pos, mOptions.mNumInitialPlays );

            newNode->mScores += scores;
            newBranch.mUct = CalculateUct( newNode, newBranchIdx );

            scores.FlipColor();
            return scores;
        }

        ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, nextBranch.mNode );

        // Accumulate the scores on our way back down the tree

        node->mScores += branchScores;
        nextBranch.mUct = CalculateUct( node, nextBranchIdx );

        branchScores.FlipColor();
        return scores;
    }

    void ProcessResult( TreeNode* node, const PlayoutJobResultRef& result, int depth = 0 )
    {
        if( node == NULL )
            return;

        MoveSpec move = result.mPathFromRoot.mMove[depth];

        int childIdx = node->FindMoveIndex( move );
        if( childIdx < 0 )
            return;

        TreeNode* child = node->mBranch[childIdx].mNode;
        if( child == NULL )
            return;

        ProcessResult( child, result, depth + 1 );

        ScoreCard scores = result.mScores;

        bool otherColor = (result.mPathFromRoot.mCount & 1) != 0;
        if( otherColor )
            scores.FlipColor();

        node->mScores += scores;
        node->mBranch[childIdx].mUct = CalculateUct( node, childIdx );
    }

    void ProcessAsyncResults()
    {
        vector< PlayoutResult > results = mCompletedJobs->PopAll();

        for( const auto& result : results )
            this->ProcessResult( mRoot, result );
    }

    void UpdateSearchTree()
    {
        MoveList pathFromRoot;

        this->ExpandAtLeaf( pathFromRoot, mRoot );
    }

    void SearchThreadProc()
    {
        while( !mExitSearch )
        {
            ProcessAsyncResults();

            for( int i = 0; i < mOptions.mTreeUpdateBatch; i++ )
                UpdateSearchTree();
        }
    }

    void StartSearching( const UciSearchConfig& config )
    {
        this->StopSearching();

        mUciConfig      = config;
        mExitSearch     = false;
        mSearchThread   = new std::thread( [] { this->SearchThreadProc(); } );
    }

    void StopSearching()
    {
        if( mSearchThread )
        {
            mExitSearch = true;

            mSearchThread->join();
            mSearchThread = NULL;
        }
    }
};


