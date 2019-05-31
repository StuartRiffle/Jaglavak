// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeNode;

struct BranchInfo
{
    TreeNode*   _Node;
    MoveSpec    _Move;
    ScoreCard   _Scores;
    double      _Prior;

#if DEBUG    
    int         _DebugLossCounter;
    char        _MoveText[MAX_MOVETEXT_LENGTH];
#endif

    BranchInfo() { memset( this, 0, sizeof( *this ) ); }
};

struct TreeLink
{
    TreeNode*           _Prev;
    TreeNode*           _Next;
};

struct TreeNode : public TreeLink
{
    BranchInfo*         _Info;
    int                 _Color;
    vector<BranchInfo>  _Branch;
    bool                _GameOver;
    ScoreCard           _GameResult;
    Position            _Pos;

    TreeNode() : _Info( NULL ) { Clear(); }
    ~TreeNode() { Clear(); }

    void InitPosition( const Position& pos, const MoveMap& moveMap, BranchInfo* info = NULL );
    void Clear();
    int FindMoveIndex( const MoveSpec& move );
    void SanityCheck();
};
