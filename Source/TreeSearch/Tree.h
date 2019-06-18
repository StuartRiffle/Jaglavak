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
    int                 _RefCount;
    BranchInfo*         _Info;
    int                 _Color;
    vector<BranchInfo>  _Branch;
    bool                _GameOver; 
    ScoreCard           _GameResult;
    Position            _Pos; 

    TreeNode() : _Info( NULL ) { Clear(); }
    ~TreeNode() { Clear(); }

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

        RefScope( TreeNode* node ) : _Node( node ) { node->_RefCount++; }
        ~RefScope() { node->_RefCount--; }
    };
};

