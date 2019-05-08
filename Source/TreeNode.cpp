// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "GlobalOptions.h"
#include "TreeNode.h"
//#include "TreeSearch.h"

void TreeNode::InitPosition( const Position& pos, const MoveMap& moveMap, BranchInfo* info )
{
    this->Clear();

    mPos = pos;
    mInfo = info;
    mColor = pos.mWhiteToMove? WHITE : BLACK;

    MoveList moveList;
    moveList.UnpackMoveMap( pos, moveMap );

    mBranch.resize( moveList.mCount );

    for( int i = 0; i < moveList.mCount; i++ )
    {
        mBranch[i].mMove = moveList.mMove[i];
#if DEBUG        
        MoveSpecToString( moveList.mMove[i], mBranch[i].mMoveText );
#endif
    }

    if (pos.mResult != RESULT_UNKNOWN )
    {
        assert( moveList.mCount == 0 );

        mGameResult.mWins[WHITE] = (pos.mResult == RESULT_WHITE_WIN);
        mGameResult.mWins[BLACK] = (pos.mResult == RESULT_BLACK_WIN);
        mGameResult.mPlays++;
        mGameOver = true;
    }
}

void TreeNode::Clear()
{
    // We should only ever be clearing leaf nodes, because of the MRU ordering

    for( auto& info : mBranch )
        assert( info.mNode == NULL );

    if( mInfo )
    {
        assert( mInfo->mNode == this );
        mInfo->mNode = NULL;
    }

    mInfo = NULL;
    mBranch.clear();
    mGameResult.Clear();
    mGameOver = false;
}

int TreeNode::FindMoveIndex( const MoveSpec& move )
{
    for( int i = 0; i < (int) mBranch.size(); i++ )
        if( mBranch[i].mMove == move )
            return( i );

    return( -1 );
}

void TreeNode::SanityCheck()
{
    for( auto& info : mBranch )
        if( info.mNode )
            assert( info.mNode->mInfo == &info );

    assert( mInfo->mNode == this );
}
