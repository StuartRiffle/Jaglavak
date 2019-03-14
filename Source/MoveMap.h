// MoveMap.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_MOVEMAP_H__
#define CORVID_MOVEMAP_H__


/// A map of valid move target squares

template< typename SIMD >
struct ALIGN_SIMD MoveMapT
{
    SIMD        mSlidingMovesNW;
    SIMD        mSlidingMovesNE;
    SIMD        mSlidingMovesSW;
    SIMD        mSlidingMovesSE;
    SIMD        mSlidingMovesN;
    SIMD        mSlidingMovesW;
    SIMD        mSlidingMovesE;
    SIMD        mSlidingMovesS;

    SIMD        mKnightMovesNNW;
    SIMD        mKnightMovesNNE;
    SIMD        mKnightMovesWNW;
    SIMD        mKnightMovesENE;
    SIMD        mKnightMovesWSW;
    SIMD        mKnightMovesESE;
    SIMD        mKnightMovesSSW;
    SIMD        mKnightMovesSSE;

    SIMD        mPawnMovesN;
    SIMD        mPawnDoublesN;
    SIMD        mPawnAttacksNE;
    SIMD        mPawnAttacksNW;

    SIMD        mCastlingMoves;
    SIMD        mKingMoves;
    SIMD        mCheckMask;

    INLINE PDECL SIMD CalcMoveTargets() const
    {
        SIMD slidingMoves   = mSlidingMovesNW | mSlidingMovesNE | mSlidingMovesSW | mSlidingMovesSE | mSlidingMovesN  | mSlidingMovesW  | mSlidingMovesE  | mSlidingMovesS;
        SIMD knightMoves    = mKnightMovesNNW | mKnightMovesNNE | mKnightMovesWNW | mKnightMovesENE | mKnightMovesWSW | mKnightMovesESE | mKnightMovesSSW | mKnightMovesSSE;
        SIMD otherMoves     = mPawnMovesN | mPawnDoublesN | mPawnAttacksNE | mPawnAttacksNW | mCastlingMoves | mKingMoves;
        SIMD targets        = (slidingMoves & mCheckMask) | knightMoves | otherMoves;

        return( targets );
    }

    INLINE PDECL SIMD IsInCheck() const
    {
        return( ~CmpEqual( mCheckMask, MaskAllSet< SIMD >() ) );
    }
};


#endif // CORVID_MOVEMAP_H__
