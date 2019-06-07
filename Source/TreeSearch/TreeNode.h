// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeNode;

struct BranchInfo
{
    TreeNode*   _Node; // 8
    double      _Prior; // 8
    ScoreCard   _Scores; // 24
    MoveSpec    _Move; // 4

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

struct TreeNode : public TreeLink // 16
{
    BranchInfo*         _Info; // 8
    int                 _Color; // 4             X in pos
    vector<BranchInfo>  _Branch; // 16 (?)
    bool                _GameOver; // 4          X in pos
    ScoreCard           _GameResult; // 12 
    Position            _Pos; // looots

    TreeNode() : _Info( NULL ) { Clear(); }
    ~TreeNode() { Clear(); }

    void InitPosition( const Position& pos, const MoveMap& moveMap, BranchInfo* info = NULL );
    void Clear();
    int FindMoveIndex( const MoveSpec& move );
    void SanityCheck();
};
