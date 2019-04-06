// TreeNode.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    char        mMoveText[MAX_MOVETEXT];

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

    bool                mGameOver;
    ScoreCard           mGameResult;


    TreeNode() : mInfo( NULL ), mNumChildren( 0 ), mGameOver( false ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL )
    {
        mPos = pos;
        mInfo = info;
        mNumChildren = 0;

        MoveMap moveMap;
        pos.CalcMoveMap( &moveMap );

        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );

        mBranch.clear();
        mBranch.resize( moveList.mCount );

        for( int i = 0; i < moveList.mCount; i++ )
        {
            mBranch[i].mMove = moveList.mMove[i];
            MoveSpecToString( moveList.mMove[i], mBranch[i].mMoveText );
        }

        mColor = pos.mWhiteToMove? WHITE : BLACK;

        int result = (int) pos.CalcGameResult( moveMap );
        if (result != RESULT_UNKNOWN )
        {
            mGameOver = true;

            if( result == RESULT_WHITE_WIN )
                mGameResult.mWins[WHITE]++;

            if( result == RESULT_BLACK_WIN )
                mGameResult.mWins[BLACK]++;

            mGameResult.mPlays++;
        }
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


