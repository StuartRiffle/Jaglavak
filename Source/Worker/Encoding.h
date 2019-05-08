// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "TreeSearch.h"

struct EncodedPosition
{
    float mPawns[64];
    float mKnights[64];
    float mBishops[64];
    float mRooks[64];
    float mQueens[64];
    float mKings[64];
};

struct EncodedMoveEval
{
    float mMoveValue[64][64];
}

static vector< float > PositionToTensor( const Position& pos )
{
    EncodedPosition encoded = { 0 };

    float whiteVal = pos.mWhiteToPlay? 1 : -1;
    float blackVal = -whiteVal;

    for( int idx = 0; idx < 64; idx++ )
    {
        u64 bit = 1ULL << idx;

        if( pos.mWhitePawns   & bit )   encoded.mPawns[idx]   = whiteVal; else
        if( pos.mWhiteKnights & bit )   encoded.mKnights[idx] = whiteVal; else
        if( pos.mWhiteBishops & bit )   encoded.mBishops[idx] = whiteVal; else
        if( pos.mWhiteRooks   & bit )   encoded.mRooks[idx]   = whiteVal; else
        if( pos.mWhiteQueens  & bit )   encoded.mQueens[idx]  = whiteVal; else
        if( pos.mWhiteKing    & bit )   encoded.mKings[idx]   = whiteVal; else
        if( pos.mBlackPawns   & bit )   encoded.mPawns[idx]   = blackVal; else
        if( pos.mBlackKnights & bit )   encoded.mKnights[idx] = blackVal; else
        if( pos.mBlackBishops & bit )   encoded.mBishops[idx] = blackVal; else
        if( pos.mBlackRooks   & bit )   encoded.mRooks[idx]   = blackVal; else
        if( pos.mBlackQueens  & bit )   encoded.mQueens[idx]  = blackVal; else
        if( pos.mBlackKing    & bit )   encoded.mKings[idx]   = blackVal;
    }

    size_t numElems = sizeof( EncodedPosition ) / sizeof( float );

    vector< float > result( &encoded, &encoded + numElems );
    return result;
}

static vector< float > ExtractMoveEval( const vector< float >& data, const MoveList& moveList )
{
    vector< float > result;
    size_t numElems = sizeof( EncodedMoveEval ) / sizeof( float );

    assert( data.size() == numElems );
    if( data.size() == numElems )
    {
        EncodedMoveEval eval;
        memcpy( &eval, data.data(), sizeof( eval ) );

        result.reserve( moveList.mCount );

        for( int i = 0; i < moveList.mCount; i++ )
        {
            const MoveSpec& spec = moveList.mMove[i];

            float eval = eval->mMoveValue( spec.mSrc, spec.mDest );
            result.push_back( eval );
        }
    }

    return result;
}

