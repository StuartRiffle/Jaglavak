// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    double      mPrior;

#if DEBUG    
    int         mDebugLossCounter;
    char        mMoveText[MAX_MOVETEXT_LENGTH];
#endif

    BranchInfo() { memset( this, 0, sizeof( *this ) ); }
};

struct TreeLink
{
    TreeNode*           mPrev;
    TreeNode*           mNext;
};

struct TreeNode : public TreeLink
{
    BranchInfo*         mInfo;
    int                 mColor;
    vector<BranchInfo>  mBranch;
    bool                mGameOver;
    ScoreCard           mGameResult;
    Position            mPos;

    TreeNode() : mInfo( NULL ) { Clear(); }
    ~TreeNode() { Clear(); }

    void InitPosition( const Position& pos, const MoveMap& moveMap, BranchInfo* info = NULL );
    void Clear();
    int FindMoveIndex( const MoveSpec& move );
    void SanityCheck();
};
