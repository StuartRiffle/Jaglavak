// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

/*
struct PackedMoveSpec
{
    union
    {
        struct
        {
            uint16_t _Src   : 6;
            uint16_t _Dest  : 6;
            uint16_t _Promo : 2;
        }

        uint16_t _Word;
    };

    INLINE PDECL operator(uint16_t)() const { return _Word; }
};
 */

/// The parameters for one move of the game
//       
template< typename T >
struct MoveSpecT
{
    T   _Src;
    T   _Dest;
    T   _Type;
    T   _Flags;

    INLINE PDECL MoveSpecT() {}
    INLINE PDECL MoveSpecT( T src, T dest, T type = MOVE ) : _Src(  src ), _Dest( dest ), _Type( type ), _Flags( 0 ) {}
    INLINE PDECL void Set(  const T& _src, const T& _dest, const T& _type = MOVE ) { _Src = _src;   _Dest = _dest;   _Type = _type;   _Flags = 0; }

    template< typename U > INLINE PDECL MoveSpecT( const MoveSpecT< U >& rhs ) : _Src( rhs._Src ), _Dest( rhs._Dest ), _Type( rhs._Type ) {}

    INLINE PDECL int  IsCapture() const       { return( ((_Type >= CAPTURE_LOSING) && (_Type <= CAPTURE_WINNING)) || ((_Type >= CAPTURE_PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN)) ); }
    INLINE PDECL int  IsPromotion() const     { return( (_Type >= PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN) ); }
    INLINE PDECL int  IsSpecial() const       { return( _Flags != 0 ); }
    INLINE PDECL void Flip()                  { _Src = FlipSquareIndex( _Src ); _Dest = FlipSquareIndex( _Dest ); }
    INLINE PDECL char GetPromoteChar() const  { return( "\0\0\0\0nbrqnbrq\0\0"[_Type] ); }

    INLINE PDECL bool operator==( const MoveSpecT& rhs ) const { return( (_Src == rhs._Src) && (_Dest == rhs._Dest) && (_Type == rhs._Type) ); }
    INLINE PDECL bool operator!=( const MoveSpecT& rhs ) const { return( (_Src != rhs._Src) || (_Dest != rhs._Dest) || (_Type != rhs._Type) ); }

    template< typename SCALARSPEC >
    INLINE PDECL void Unpack( const SCALARSPEC* moves )
    {
        const int LANES = SimdWidth< T >::LANES;
        MoveSpecT< u64 > ALIGN_SIMD unpacked[LANES];

        for( int i = 0; i < LANES; i++ )
        {
            unpacked[i]._Src   = moves[i]._Src;
            unpacked[i]._Dest  = moves[i]._Dest;
            unpacked[i]._Type  = moves[i]._Type;
            unpacked[i]._Flags = moves[i]._Flags;
        }

        Swizzle< T >( unpacked, this );
    }

};


/// A list of valid moves
//
struct MoveList
{
    int         _Count;
    MoveSpec    _Move[MAX_POSSIBLE_MOVES];

    INLINE PDECL      MoveList()                      { this->Clear(); }
    INLINE PDECL void Clear()                         { _Count = 0; }
    INLINE PDECL void FlipAll()                       { for( int i = 0; i < _Count; i++ ) _Move[i].Flip(); }
    INLINE PDECL void Append( const MoveSpec& spec )  { _Move[_Count++] = spec; }

    PDECL int LookupMove( const MoveSpec& spec )
    {
        for( int idx = 0; idx < _Count; idx++ )
            if( (_Move[idx]._Src == spec._Src) && (_Move[idx]._Dest == spec._Dest) )
                if( _Move[idx].GetPromoteChar() == spec.GetPromoteChar() )
                    return idx;

        return -1;
    }

    PDECL MoveSpec Remove( int idx )
    {
        MoveSpec result = _Move[idx];
        _Move[idx] = _Move[--_Count];
        return result;
    }

    PDECL void DiscardMovesBelow( int type )
    {
        int prevCount = _Count;

        for( _Count = 0; _Count < prevCount; _Count++ )
            if( _Move[_Count]._Type < type )
                break;

        for( int idx = _Count + 1; idx < prevCount; idx++ )
            if( _Move[idx]._Type >= type )
                _Move[_Count++] = _Move[idx];
    }

    PDECL void DiscardQuietMoves()
    {
        this->DiscardMovesBelow( CAPTURE_EQUAL );
    }

    PDECL void UnpackMoveMap( const Position& pos, const MoveMap& mmap )
    {
        this->Clear();

        u64 whitePieces = pos._WhitePawns | pos._WhiteKnights | pos._WhiteBishops | pos._WhiteRooks | pos._WhiteQueens | pos._WhiteKing;

        if( mmap._PawnMovesN )      this->StorePawnMoves( pos, mmap._PawnMovesN,     SHIFT_N            );
        if( mmap._PawnDoublesN )    this->StorePawnMoves( pos, mmap._PawnDoublesN,   SHIFT_N * 2        );
        if( mmap._PawnAttacksNE )   this->StorePawnMoves( pos, mmap._PawnAttacksNE,  SHIFT_NE           );
        if( mmap._PawnAttacksNW )   this->StorePawnMoves( pos, mmap._PawnAttacksNW,  SHIFT_NW           );
        if( mmap._CastlingMoves )   this->StoreKingMoves( pos, mmap._CastlingMoves,  pos._WhiteKing     );
        if( mmap._KingMoves )       this->StoreKingMoves( pos, mmap._KingMoves,      pos._WhiteKing     );

        if( mmap._SlidingMovesNW )  this->StoreSlidingMoves< SHIFT_NW >( pos, mmap._SlidingMovesNW, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesNE )  this->StoreSlidingMoves< SHIFT_NE >( pos, mmap._SlidingMovesNE, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesSW )  this->StoreSlidingMoves< SHIFT_SW >( pos, mmap._SlidingMovesSW, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesSE )  this->StoreSlidingMoves< SHIFT_SE >( pos, mmap._SlidingMovesSE, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesN  )  this->StoreSlidingMoves< SHIFT_N  >( pos, mmap._SlidingMovesN,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesW  )  this->StoreSlidingMoves< SHIFT_W  >( pos, mmap._SlidingMovesW,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesE  )  this->StoreSlidingMoves< SHIFT_E  >( pos, mmap._SlidingMovesE,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesS  )  this->StoreSlidingMoves< SHIFT_S  >( pos, mmap._SlidingMovesS,  whitePieces, mmap._CheckMask );

        if( mmap._KnightMovesNNW )  this->StoreStepMoves( pos, mmap._KnightMovesNNW, SHIFT_N + SHIFT_NW );
        if( mmap._KnightMovesNNE )  this->StoreStepMoves( pos, mmap._KnightMovesNNE, SHIFT_N + SHIFT_NE );
        if( mmap._KnightMovesWNW )  this->StoreStepMoves( pos, mmap._KnightMovesWNW, SHIFT_W + SHIFT_NW );
        if( mmap._KnightMovesENE )  this->StoreStepMoves( pos, mmap._KnightMovesENE, SHIFT_E + SHIFT_NE );
        if( mmap._KnightMovesWSW )  this->StoreStepMoves( pos, mmap._KnightMovesWSW, SHIFT_W + SHIFT_SW );
        if( mmap._KnightMovesESE )  this->StoreStepMoves( pos, mmap._KnightMovesESE, SHIFT_E + SHIFT_SE );
        if( mmap._KnightMovesSSW )  this->StoreStepMoves( pos, mmap._KnightMovesSSW, SHIFT_S + SHIFT_SW );
        if( mmap._KnightMovesSSE )  this->StoreStepMoves( pos, mmap._KnightMovesSSE, SHIFT_S + SHIFT_SE );

        if( pos._BoardFlipped )
            this->FlipAll();
    }

    PDECL int FindMoves( const Position& pos )
    {
        MoveMap mmap;

        this->Clear();
        pos.CalcMoveMap( &mmap );
        this->UnpackMoveMap( pos, mmap );

        return( this->_Count );
    }

private:
    INLINE PDECL void ClassifyAndStoreMove( const Position& pos, int srcIdx, int destIdx, int promote = 0 ) 
    {
        u64 src         = SquareBit( (u64) srcIdx );
        u64 dest        = SquareBit( (u64) destIdx );
        int src_val     = (src  & pos._WhitePawns)? 1 : ((src  & (pos._WhiteKnights | pos._WhiteBishops))? 3 : ((src  & pos._WhiteRooks)? 5 : ((src  & pos._WhiteQueens)? 9 : 20)));
        int dest_val    = (dest & pos._BlackPawns)? 1 : ((dest & (pos._BlackKnights | pos._BlackBishops))? 3 : ((dest & pos._BlackRooks)? 5 : ((dest & pos._BlackQueens)? 9 :  0)));
        int relative    = SignOrZero( dest_val - src_val );
        int capture     = dest_val? (relative + 2) : 0;
        int type        = promote? (promote + (capture? 4 : 0)) : capture;

        _Move[_Count++].Set( srcIdx, destIdx, type );
    }

    PDECL void StorePromotions( const Position& pos, u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );

            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_QUEEN  );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_ROOK   );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_BISHOP );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx, PROMOTE_KNIGHT );
        }
    }

    INLINE PDECL void StoreStepMoves( const Position& pos, u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->ClassifyAndStoreMove( pos, idx - ofs, idx );
        }
    }

    template< int SHIFT >
    INLINE PDECL void StoreSlidingMoves( const Position& pos, u64 dest, u64 pieces, u64 checkMask ) 
    {
        u64 src = Shift< -SHIFT >( dest ) & pieces;
        u64 cur = Shift<  SHIFT >( src );
        int ofs = SHIFT;

        while( cur )
        {
            this->StoreStepMoves( pos, cur & checkMask, ofs );
            cur = Shift< SHIFT >( cur ) & dest;
            ofs += SHIFT;
        }
    }

    PDECL void StorePawnMoves( const Position& pos, u64 dest, int ofs ) 
    {
        this->StoreStepMoves(  pos, dest & ~RANK_8, ofs );
        this->StorePromotions( pos, dest &  RANK_8, ofs );
    }

    PDECL void StoreKingMoves( const Position& pos, u64 dest, u64 king ) 
    {
        int kingIdx = (int) LowestBitIndex( king );

        do
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->ClassifyAndStoreMove( pos, kingIdx, idx );
        }
        while( dest );
    }
};
