// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "SearchTree.h"
#include "Util/FEN.h"

void SearchTree::Init()
{
    _NodePoolEntries = _Settings->Get( "Search.NumTreeNodes" );

    size_t totalSize = _NodePoolEntries * sizeof( TreeNode );
    _NodePoolBuf = unique_ptr< HugeBuffer >( new HugeBuffer( totalSize ) );
    _NodePool = (TreeNode*) _NodePoolBuf->_Ptr;

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

void SearchTree::Touch( TreeNode* node )
{
    // Move this node to the front of the MRU list

    assert( node->_Next->_Prev == node );
    assert( node->_Prev->_Next == node );

    node->_Next->_Prev = node->_Prev;
    node->_Prev->_Next = node->_Next;

    node->_Next = _MruListHead._Next;
    node->_Next->_Prev = node;

    node->_Prev = (TreeNode*) &_MruListHead;
    node->_Prev->_Next = node;
}

TreeNode* SearchTree::CreateBranch( TreeNode* node, int branchIdx )
{
    TreeNode* newNode = AllocNode();
    assert( newNode != node );

    BranchInfo* chosenBranch = &node->_Branch[branchIdx];
    assert( chosenBranch->_Node == NULL );

    MoveMap newMap;
    Position newPos = node->_Pos;
    newPos.Step( chosenBranch->_Move, &newMap );

    ClearNode( newNode );
    InitNode( newNode, newPos, newMap, chosenBranch ); 
//    EstimatePriors( newNode );

    chosenBranch->_Node = newNode;
    return newNode;
}

TreeNode* SearchTree::AllocNode()
{
    TreeNode* node = _MruListHead._Prev;

    ClearNode( node );
    Touch( node );

    //_Metrics._NumNodesCreated++;
    return node;
}

void SearchTree::InitNode( TreeNode* node, const Position& pos, const MoveMap& moveMap, BranchInfo* info )
{
    node->_Pos = pos;
    node->_Info = info;
    node->_Color = pos._WhiteToMove? WHITE : BLACK;

    if (pos._GameResult != RESULT_UNKNOWN )
    {
        node->_GameResult._Wins[WHITE] = (pos._GameResult == RESULT_WHITE_WIN);
        node->_GameResult._Wins[BLACK] = (pos._GameResult == RESULT_BLACK_WIN);
        node->_GameResult._Plays = 1;
        node->_GameOver = true;
    }
    else
    {
        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );

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

void SearchTree::ClearNode( TreeNode* node )
{
    // Make sure we don't delete any nodes that are still being used by a fiber.
    // This should never actually happen, because the tree is huge.

    while( node->_RefCount > 0 )
    {
        // -----------------------------------------------------------------------------------
        YIELD_FIBER();
        // -----------------------------------------------------------------------------------
    }

    // We should only ever see leaf nodes at the end of the MRU list.
    // Anything else indicates a bug.

    for( auto& info : node->_Branch )
    {
        assert( info._Node == NULL );
    }

    // Detach from the parent branch

    if( node->_Info )
    {
        assert( node->_Info->_Node == node );
        node->_Info->_Node = NULL;
    }

    // Reset for the next user

    node->_Info = NULL;
    node->_Branch.clear();
    node->_GameResult.Clear();
    node->_GameOver = false;
    node->_RefCount = 0;
}

void SearchTree::SetPosition( const Position& pos )
{
    MoveMap moveMap;
    pos.CalcMoveMap( &moveMap );

    if( _SearchRoot )
        _SearchRoot->_Info = NULL;

    _SearchRoot = AllocNode();
    _SearchRoot->_Info = &_RootInfo;
    _RootInfo._Node = _SearchRoot;

    InitNode( _SearchRoot, pos, moveMap, _SearchRoot->_Info );
}

