// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

/// A map of valid move target squares

template< typename SIMD >
struct ALIGN_SIMD MoveMapT
{
    SIMD    _SlidingMovesN;
    SIMD    _SlidingMovesNW;
    SIMD    _SlidingMovesW;
    SIMD    _SlidingMovesSW;
    SIMD    _SlidingMovesS;
    SIMD    _SlidingMovesSE;
    SIMD    _SlidingMovesE;
    SIMD    _SlidingMovesNE;

    SIMD    _KnightMovesNNW;
    SIMD    _KnightMovesNNE;
    SIMD    _KnightMovesWNW;
    SIMD    _KnightMovesENE;
    SIMD    _KnightMovesWSW;
    SIMD    _KnightMovesESE;
    SIMD    _KnightMovesSSW;
    SIMD    _KnightMovesSSE;

    SIMD    _PawnMovesN;
    SIMD    _PawnDoublesN;
    SIMD    _PawnAttacksNE;
    SIMD    _PawnAttacksNW;
    
    SIMD    _CastlingMoves;
    SIMD    _KingMoves;
    SIMD    _CheckMask;

    INLINE PDECL SIMD CalcMoveTargets() const
    {
        SIMD slidingMoves   = _SlidingMovesNW | _SlidingMovesNE | _SlidingMovesSW | _SlidingMovesSE | _SlidingMovesN  | _SlidingMovesW  | _SlidingMovesE  | _SlidingMovesS;
        SIMD knightMoves    = _KnightMovesNNW | _KnightMovesNNE | _KnightMovesWNW | _KnightMovesENE | _KnightMovesWSW | _KnightMovesESE | _KnightMovesSSW | _KnightMovesSSE;
        SIMD otherMoves     = _PawnMovesN | _PawnDoublesN | _PawnAttacksNE | _PawnAttacksNW | _CastlingMoves | _KingMoves;
        SIMD targets        = (slidingMoves & _CheckMask) | knightMoves | otherMoves;

        return( targets );
    }

    INLINE PDECL SIMD IsInCheck() const
    {
        return( ~CmpEqual( _CheckMask, MaskAllSet< SIMD >() ) );
    }
};

