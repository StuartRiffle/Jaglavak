// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"

void TreeNode::InitPosition( const Position& pos, const MoveMap& moveMap, BranchInfo* info )
{
    this->Clear();

    _Pos = pos;
    _Info = info;
    _Color = pos._WhiteToMove? WHITE : BLACK;

    MoveList moveList;
    moveList.UnpackMoveMap( pos, moveMap );

    if (pos._GameResult != RESULT_UNKNOWN )
    {
        assert( moveList._Count == 0 );

        _GameResult._Wins[WHITE] = (pos._GameResult == RESULT_WHITE_WIN);
        _GameResult._Wins[BLACK] = (pos._GameResult == RESULT_BLACK_WIN);
        _GameResult._Plays = 1;
        _GameOver = true;
    }
    else
    {
        _Branch.resize( moveList._Count );

        for( int i = 0; i < moveList._Count; i++ )
        {
            _Branch[i]._Move = moveList._Move[i];
#if DEBUG        
            MoveSpecToString( moveList._Move[i], _Branch[i]._MoveText );
#endif
        }
    }
}

void TreeNode::Clear()
{
    // We should only ever be clearing leaf nodes, because of the MRU ordering

    for( auto& info : _Branch )
        assert( info._Node == NULL );

    if( _Info )
    {
        assert( _Info->_Node == this );
        _Info->_Node = NULL;
    }

    _Info = NULL;
    _Branch.clear();
    _GameResult.Clear();
    _GameOver = false;
}

int TreeNode::FindMoveIndex( const MoveSpec& move )
{
    for( int i = 0; i < (int) _Branch.size(); i++ )
        if( _Branch[i]._Move == move )
            return( i );

    return( -1 );
}

void TreeNode::SanityCheck()
{
    for( auto& info : _Branch )
        if( info._Node )
            assert( info._Node->_Info == &info );

    assert( _Info->_Node == this );
}
