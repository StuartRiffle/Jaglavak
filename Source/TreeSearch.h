struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    float       mUCT;

    BranchInfo()
    {
        mNode   = NULL;
        mUCT    = 1.0f;
    }
};


    /*
float CalcUct(BranchInfo& branch, int childIndex)
{
    BranchInfo& child = branch.children[childIndex];

    float childWinRatio = child.wins * 1.0f / child.plays;
    float uct = childWinRatio + exploration * sqrtf(logf(branch.plays) / child.plays);

    return uct;
}
*/


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


struct SearchOptions
{
    int mMaxTreeNodes;
    int mNumInitialPlays;
    int mNumAsyncPlays;

    PlayoutOptions mPlayout;
};


typedef list< TreeNode > TreeNodeList;

struct SearchTree
{
    TreeNode*       mRoot;
    TreeNodeList    mNodes;
    SearchOptions*  mOptions;

    SearchTree( SearchOptions* options, const char* fen = NULL )
    {
        mOptions = options;

        Position pos;
        pos.Reset();

        if( fen )
            FEN::StringToPosition( fen, &mRoot->mPos );

        mRoot = AllocNode();
        mRoot->Init( pos );
    }

    ~SearchTree()
    {
        DeleteAllNodes();
    }

    void MoveToFront( TreeNodeList::iterator iter )
    {
        mNodes.splice( mNodes.begin(), mNodes, iter );
    }

    TreeNode* AllocNode()
    {
        int limit = mOptions->mNodeLimit;

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

    typedef std::shared_ptr< PlayoutJobInfo >   PlayoutJobInfoRef;
    typedef std::shared_ptr< PlayoutJobResult > PlayoutJobResultRef;

    typedef ThreadSafeQueue< PlayoutJobInfoRef >   PlayoutJobQueue;
    typedef ThreadSafeQueue< PlayoutJobResultRef > PlayoutResultQueue;

    ScoreCard PlaySomeGames( MoveList& pathFromRoot, BranchInfo& info )
    {
        Position pos = info.mPos;
        pos.ApplyMove( info.mMove );

        // Queue some bulk async playouts

        PlayoutJobInfo job;

        job.mPosition = pos;
        job.mOptions = mOptions->mPlayout;
        job.mNumGames = mOptions->mNumAsyncPlays;
        job.mPathFromRoot.assign( pathFromRoot.Moves, pathFromRoot.mMoves + pathFromRoot.mCount );

        float gamePhase = Evaluation::CalcGamePhase( pos, mOptions->mUsePopcnt );
        Evaluation::GenerateWeights( &job.mWeights, gamePhase );

        mPlayoutJobQueue.Push( job );

        // Also do a more modest batch now

        PlayoutProvider provider( &mOptions->mPlayoutOptions );
        ScoreCard scores = provider.Play( pos, mOptions->mNumInitialPlays );

        return scores;
    }



    ScoreCard ExpandAtLeaf( TreeNode* node )
    {
        MoveToFront( node );

        int idx = SelectNextBranch( node );
        BranchInfo& info = mNode->mBranch[idx];

        pathFromRoot.Add( info.mMove );

        if( !info.mNode )
        {
            TreeNode* child = AllocNode();
            child->Init( node->mPos, &info ); 

            info.mNode = child;
            node->mChildren++;
           
            int randomIdx = mOptions->mRandom.Get( child->mBranch.size() );

            BranchInfo& newBranch = child->mBranch[randomIdx];
            pathFromRoot.Add( newBranch.mMove );

            ScoreCard scores = PlaySomeGames( pathFromRoot, newBranch );
            return scores;
        }

        ScoreCard scores = ExpandAtLeaf( pathFromRoot, info.mNode );
        info.mNode->mScore += scores;
        // TODO recalc UCT

        return scores;
    }

    void ProcessPlayoutResult( const PlayoutResult& result )
    {

            for( int i = 0; i < job->mMoveList, i++ )
            {

                // TODO update the damned counts

                int nextIdx = node->FindMoveIndex( move );
                if( nextIdx < 0 )
                    break;

                node = node->mMoves[nextIdx];
            }
        }
    }

    void Step()
    {
        if( mJobQueue.IsFull() )
            std::thread::yield();

        if( !mJobQueue.IsFull() )
        {
            MoveList pathFromRoot;
            this->ExpandAtLeaf( pathFromRoot, mRoot );
        }

        vector< PlayoutResult > results = mCompletedJobs->PopAll();

        for( auto& result : results )
            this->ProcessPlayoutResult( result );
    }


};


