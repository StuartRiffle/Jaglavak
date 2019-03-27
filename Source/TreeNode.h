// TreeNode.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;

    BranchInfo() : mNode( NULL ) {}
};

struct TreeLink
{
    TreeNode*           mPrev;
    TreeNode*           mNext;
};

struct TreeNode : public TreeLink
{
    Position            mPos;
    BranchInfo*         mInfo;
    int                 mColor;
    int                 mNumChildren;
    std::vector<BranchInfo>  mBranch;

    TreeNode() : mInfo( NULL ), mNumChildren( 0 ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL )
    {
        mPos = pos;
        mInfo = info;
        mNumChildren = 0;

        MoveList moveList;
        moveList.FindMoves( pos );

        mBranch.clear();
        mBranch.resize( moveList.mCount );

        for( int i = 0; i < moveList.mCount; i++ )
            mBranch[i].mMove = moveList.mMove[i];

        mColor = (int) pos.mWhiteToMove;
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


