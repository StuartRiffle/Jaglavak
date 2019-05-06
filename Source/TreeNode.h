// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    float       mPrior;

#if DEBUG    
    char        mMoveText[MAX_MOVETEXT_LENGTH];
#endif

    BranchInfo() { memset( this, 0, sizeof( this ) ); }
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

    bool                mGameOver;
    ScoreCard           mGameResult;

    TreeNode() : mInfo( NULL ), mGameOver( false ), mCounter( -1 ), mTouch( 0 ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL );
    void Clear();
    int FindMoveIndex( const MoveSpec& move );
    void SanityCheck();
};
