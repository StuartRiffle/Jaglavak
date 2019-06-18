// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"

void Tree::CreateNodePool()
{
    _NodePoolEntries = _Options->_MaxTreeNodes;
    _NodePoolBuf.resize( _NodePoolEntries * sizeof( TreeNode ) + SIMD_ALIGNMENT );
    _NodePool = (TreeNode*) (((uintptr_t) _NodePoolBuf.data() + SIMD_ALIGNMENT - 1) & ~(SIMD_ALIGNMENT - 1));

    for( int i = 0; i < _NodePoolEntries; i++ )
    {
        _NodePool[i]._Prev = &_NodePool[i - 1];
        _NodePool[i]._Next = &_NodePool[i + 1];
    }

    _NodePool[0]._Prev = (TreeNode*) &_MruListHead;
    _MruListHead._Next = &_NodePool[0];

    _NodePool[_NodePoolEntries - 1]._Next = (TreeNode*) &_MruListHead;
    _MruListHead._Prev = &_NodePool[_NodePoolEntries - 1];
}

void Tree::MoveToFront( TreeNode* node )
{
    assert( node->_Next->_Prev == node );
    assert( node->_Prev->_Next == node );

    node->_Next->_Prev = node->_Prev;
    node->_Prev->_Next = node->_Next;

    node->_Next = _MruListHead._Next;
    node->_Next->_Prev = node;

    node->_Prev = (TreeNode*) &_MruListHead;
    node->_Prev->_Next = node;
}

void Tree::CreateNode( TreeNode* node, int branchIdx )
{
    TreeNode* newNode = AllocNode();
    assert( newNode != node );

    BranchInfo* chosenBranch = &node->_Branch[branchIdx];

    MoveToFront( node );

    MoveMap newMap;
    Position newPos = node->_Pos;

    newPos.Step( chosenBranch->_Move, &newMap );

    ClearNode( newNode );
    InitNode( newNode, newPos, newMap, chosenBranch ); 
    CalculatePriors( newNode );

    chosenBranch->_Node = newNode;
}

TreeNode* Tree::AllocNode()
{
    TreeNode* node = _MruListHead._Prev;

    ClearNode( node );
    MoveToFront( node );

    _Metrics._NumNodesCreated++;
    return node;
}

void Tree::InitNode( TreeNode* node, const Position& pos, const MoveMap& moveMap, BranchInfo* info )
{
    node->_Pos = pos;
    node->_Info = info;
    node->_Color = pos._WhiteToMove? WHITE : BLACK;

    MoveList moveList;
    moveList.UnpackMoveMap( pos, moveMap );

    if (pos._GameResult != RESULT_UNKNOWN )
    {
        assert( moveList._Count == 0 );

        node->_GameResult._Wins[WHITE] = (pos._GameResult == RESULT_WHITE_WIN);
        node->_GameResult._Wins[BLACK] = (pos._GameResult == RESULT_BLACK_WIN);
        node->_GameResult._Plays = 1;
        node->_GameOver = true;
    }
    else
    {
        node->_Branch.resize( moveList._Count );

        for( int i = 0; i < moveList._Count; i++ )
        {
            node->_Branch[i]._Move = moveList._Move[i];
#if DEBUG        
            MoveSpecToString( moveList._Move[i], node->_Branch[i]._MoveText );
#endif
        }
    }
}

void Tree::ClearNode( TreeNode* node )
{
    // This should never happen in practice, because nodes that are currently
    // being used will be near the front of the MRU list, and the tree is huge.

    assert( node->_RefCount == 0 );
    while( node->_RefCount > 0 )
        YIELD_FIBER();

    // We should only ever see leaf nodes here, because of the MRU ordering.

    for( auto& info : node->_Branch )
        assert( info._Node == NULL );

    // Detach from parent branch

    if( node->_Info )
    {
        assert( node->_Info->_Node == this );
        node->_Info->_Node = NULL;
    }

    // Reset for the next user

    _Info = NULL;
    _Branch.clear();
    _GameResult.Clear();
    _GameOver = false;
    _RefCount = 0;
}


