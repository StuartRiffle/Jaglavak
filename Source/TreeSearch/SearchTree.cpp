// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "SearchTree.h"
#include "Util/FEN.h"

void SearchTree::Init()
{
    _NodePoolEntries = _Settings->Get( "Search.MaxTreeNodes" );

    size_t totalSize = _NodePoolEntries * sizeof( TreeNode );
    _NodePoolBuf = unique_ptr< HugeBuffer >( new HugeBuffer( totalSize ) );
    _NodePool = (TreeNode*) _NodePoolBuf->_Ptr;

    _MruListHead._Next = (TreeNode*) &_MruListHead;
    _MruListHead._Prev = (TreeNode*) &_MruListHead;
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

TreeNode* SearchTree::AllocNode()
{
    if( _NodePoolUsed < _NodePoolEntries )
    {
        size_t idx = _NodePoolUsed++;

        TreeNode* newNode = _NodePool + idx;
        new(newNode) TreeNode();

        newNode->_Next = (TreeNode*) &_MruListHead;
        newNode->_Prev = newNode->_Next->_Prev;

        newNode->_Next->_Prev = newNode;
        newNode->_Prev->_Next = newNode;

        assert( _MruListHead._Prev == newNode );
    }

    TreeNode* node = _MruListHead._Prev;

    ClearNode( node );
    Touch( node );

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

    // -----------------------------------------------------------------------------------
    //FIBER_YIELD_UNTIL( node->_RefCount == 0 );
    assert( node->_RefCount == 0 );
    // -----------------------------------------------------------------------------------

    // We should only ever see leaf nodes at the end of the MRU list.
    // Anything else is a bug.

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

void SearchTree::EstimatePriors( TreeNode* node )
{
    // TODO
    for( auto& info : node->_Branch )
        info._Prior = 0;
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

    InitNode( newNode, newPos, newMap, chosenBranch );
    EstimatePriors( newNode );

    chosenBranch->_Node = newNode;
    return newNode;
}


void SearchTree::SetPosition( const Position& startPos )
{
    Position pos = startPos;

    MoveMap moveMap;
    pos.CalcMoveMap( &moveMap );

    if( _SearchRoot )
        _SearchRoot->_Info = NULL;

    _SearchRoot = AllocNode();
    _SearchRoot->_Info = &_RootInfo;
    _RootInfo._Node = _SearchRoot;

    InitNode( _SearchRoot, pos, moveMap, _SearchRoot->_Info );
}



void  SearchTree::VerifyTopology() const
{
#if DEBUG
    set< TreeNode* > deletedNodes;
    u64 highestTouch = 0;

    TreeNode* node = (TreeNode*) _MruListHead._Prev;
    while(node != _MruListHead._Next)
    {
        assert( node->_TouchSerial >= highestTouch );
        highestTouch = node->_TouchSerial;

        if( node->_Info )
        {
            assert( node->_Info->_Node == node );
        }

        if( node != _SearchRoot )
        {
            for( auto& info : node->_Branch )
            {
                if( info._Node != NULL )
                {
                    assert( deletedNodes.find( info._Node ) != deletedNodes.end() );
                }
            }
        }

        deletedNodes.insert( node );
        node = (TreeNode*) node->_Prev;
    }         
#endif
}

void SearchTree::Dump( TreeNode* node, int depth, int topMoves, string prefix ) const
{
    if( node == NULL )
        return;

    u64 totalPlayed = 0;
    for( auto& info : node->_Branch )
        totalPlayed += info._Scores._Plays;

    if( totalPlayed == 0 )
        return;

    vector< pair< u64, int > > playsByIndex;
    int idx = 0;
    for( auto& info : node->_Branch )
        playsByIndex.emplace_back( info._Scores._Plays, idx++ );

    sort(    playsByIndex.begin(), playsByIndex.end() );
    reverse( playsByIndex.begin(), playsByIndex.end() );

    int movesToShow = topMoves;
    if( movesToShow == 0 )
        movesToShow = (int) node->_Branch.size();

    int movesShown = 0;
    for( auto iter : playsByIndex )
    {
        cout << "info string " << prefix;

        u64 plays = iter.first;
        int moveIndex = iter.second;

        assert( moveIndex < node->_Branch.size() );
        BranchInfo& info = node->_Branch[moveIndex];

        string moveText = SerializeMoveSpec( info._Move );
        cout << moveText << " ";

        float frac = info._Scores._Plays * 1.0f / totalPlayed;
        float pct = frac * 100;

        cout << frac << " (" << plays << ")" << endl;

        if( depth > 1 )
            this->Dump( info._Node, depth - 1, topMoves, prefix + "     " );       

        if( ++movesShown >= movesToShow )
            break;
    }


}

void SearchTree::DumpRoot() const
{
    this->Dump( _SearchRoot, 1, 0 );
}

void SearchTree::DumpTop() const
{
    this->Dump( _SearchRoot, 3, 3 );
}

