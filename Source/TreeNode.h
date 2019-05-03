// TreeNode.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    char        mMoveText[MAX_MOVETEXT];

    BranchInfo() : mNode( NULL ) {}
};

struct ALIGN_SIMD TreeLink
{
    TreeNode*           mPrev;
    TreeNode*           mNext;
};

struct TreeNode : public TreeLink
{
    Position            mPos;
    BranchInfo*         mInfo;
    int                 mColor;
    std::vector<BranchInfo>  mBranch;
    u64                 mCounter;
    int                 mTouch;

    bool                mGameOver;
    ScoreCard           mGameResult;


    TreeNode() : mInfo( NULL ), mGameOver( false ), mCounter( -1 ), mTouch( 0 ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL )
    {
        static u64 sCount = 1;
        mCounter = sCount++;

        mPos = pos;
        mInfo = info;

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

        mGameOver = false;
        mGameResult.Clear();

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
        for( auto& info : mBranch )
            assert( info.mNode == NULL );

        if( mInfo )
        {
            assert( mInfo->mNode == this );
            mInfo->mNode = NULL;
        }

        mInfo = NULL;
        mBranch.clear();
    }

    int FindMoveIndex( const MoveSpec& move )
    {
        for( int i = 0; i < (int) mBranch.size(); i++ )
            if( mBranch[i].mMove == move )
                return( i );

        return( -1 );
    }

    void SanityCheck()
    {
        for( auto& info : mBranch )
            if( info.mNode )
                assert( info.mNode->mInfo == &info );

        assert( mInfo->mNode == this );

    }
};


