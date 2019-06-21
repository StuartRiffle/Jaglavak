// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "HugeBuffer.h"

struct TreeNode;

struct BranchInfo
{
    TreeNode*   _Node = NULL;
    ScoreCard   _Scores;
    MoveSpec    _Move;

    float       _Prior = 0;
    float       _VirtualLoss = 0;

#if DEBUG    
    char        _MoveText[MAX_MOVETEXT_LENGTH];
#endif
};

struct TreeLink
{
    TreeNode*           _Prev;
    TreeNode*           _Next;
};

struct TreeNode : public TreeLink
{
    int                 _RefCount = 0;
    BranchInfo*         _Info = NULL;
    int                 _Color = 0;
    vector<BranchInfo>  _Branch;
    bool                _GameOver = false; 
    ScoreCard           _GameResult;
    Position            _Pos; 

    TreeNode() : _RefCount( 0 ), _Info( NULL ), _GameOver( false ) { }
    ~TreeNode() {}

    int FindMoveIndex( const MoveSpec& move )
    {
        for( int i = 0; i < (int) _Branch.size(); i++ )
            if( _Branch[i]._Move == move )
                return( i );

        return( -1 );
    }

    void SanityCheck()
    {
        for( auto& info : _Branch )
            if( info._Node )
                assert( info._Node->_Info == &info );

        assert( _Info->_Node == this );
    }

    struct RefScope
    {
        TreeNode* _Node;

        RefScope( TreeNode* node ) : _Node( node ) { _Node->_RefCount++; }
        ~RefScope() { _Node->_RefCount--; }
    };
};

class SearchTree
{
    GlobalSettings*  _Settings;

    TreeNode*       _NodePool = NULL;
    size_t          _NodePoolEntries = 0;
    unique_ptr< HugeBuffer > _NodePoolBuf;

    TreeNode*       _SearchRoot = NULL;
    TreeLink        _MruListHead;
    BranchInfo      _RootInfo;

    TreeNode* AllocNode();
    void InitNode( TreeNode* node, const Position& pos, const MoveMap& moveMap, BranchInfo* info );
    void ClearNode( TreeNode* node );

public:
    SearchTree( GlobalSettings* settings ) : _Settings( settings ) {}

    void Init();
    void SetPosition( const Position& pos );
    TreeNode* CreateBranch( TreeNode* node, int branchIdx );
    void Touch( TreeNode* node );

    TreeNode* GetRootNode() { return _SearchRoot; }
};






