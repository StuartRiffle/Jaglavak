// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

/// The parameters for one move of the game
//       
template< typename T >
struct MoveSpecT
{
    T   _Src;
    T   _Dest;
    T   _Type;

    INLINE PDECL MoveSpecT() {}
    INLINE PDECL MoveSpecT( T src, T dest, T type = 0 ) : _Src(  src ), _Dest( dest ), _Type( type ) {}
    INLINE PDECL void Set(  T src, T dest, T type = 0 ) { _Src = src;  _Dest = dest;   _Type = type; }

    template< typename U > INLINE PDECL MoveSpecT( const MoveSpecT< U >& rhs ) : _Src( rhs._Src ), _Dest( rhs._Dest ), _Type( rhs._Type ) {}

    INLINE PDECL int  IsCapture() const       { return( ((_Type >= CAPTURE_LOSING) && (_Type <= CAPTURE_WINNING)) || ((_Type >= CAPTURE_PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN)) ); }
    INLINE PDECL int  IsPromotion() const     { return( (_Type >= PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN) ); }
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
        }

        Swizzle< T >( unpacked, this );
    }

    INLINE PDECL int GetAsInt() const
    {
        int result = 0;
        assert( sizeof( *this ) <= sizeof( int ) );
        
        *((MoveSpecT*) &result) = *this;
        return result;
    }

    INLINE PDECL void SetFromInt( int val )
    {
        assert( sizeof( *this ) <= sizeof( int ) );
        *this = *((MoveSpecT*) &val);
    }

    INLINE PDECL bool operator<( const MoveSpecT& other ) const
    {
        return( memcmp( this, &other, sizeof( *this ) ) == 0 );
    }
};


/// A list of valid moves
//
struct MoveList
{
    int         _Count;
    MoveSpec    _Move[MAX_POSSIBLE_MOVES];

    INLINE PDECL MoveList()     { this->Clear(); }
    INLINE PDECL void Clear()   { _Count = 0; }
    INLINE PDECL void FlipAll() { for( int i = 0; i < _Count; i++ ) _Move[i].Flip(); }
    INLINE PDECL void Append( const MoveSpec& spec )  { _Move[_Count++] = spec; }

    INLINE PDECL MoveList( const MoveList& rhs )
    {
        _Count = rhs._Count;
        for( int i = 0; i < _Count; i++ )
            _Move[i] = rhs._Move[i];
    }

    PDECL MoveSpec Remove( int idx )
    {
        MoveSpec result = _Move[idx];
        _Move[idx] = _Move[--_Count];
        return result;
    }

    PDECL void UnpackMoveMap( const Position& pos, const MoveMap& mmap )
    {
        this->Clear();

        u64 whitePieces = pos._WhitePawns | pos._WhiteKnights | pos._WhiteBishops | pos._WhiteRooks | pos._WhiteQueens | pos._WhiteKing;

        if( mmap._PawnMovesN )      this->StorePawnMoves( mmap._PawnMovesN,     SHIFT_N );
        if( mmap._PawnDoublesN )    this->StorePawnMoves( mmap._PawnDoublesN,   SHIFT_N * 2 );
        if( mmap._PawnAttacksNE )   this->StorePawnMoves( mmap._PawnAttacksNE,  SHIFT_NE );
        if( mmap._PawnAttacksNW )   this->StorePawnMoves( mmap._PawnAttacksNW,  SHIFT_NW );
        if( mmap._CastlingMoves )   this->StoreKingMoves( mmap._CastlingMoves,  pos._WhiteKing );
        if( mmap._KingMoves )       this->StoreKingMoves( mmap._KingMoves,      pos._WhiteKing );

        if( mmap._SlidingMovesNW )  this->StoreSlidingMoves< SHIFT_NW >( mmap._SlidingMovesNW, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesNE )  this->StoreSlidingMoves< SHIFT_NE >( mmap._SlidingMovesNE, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesSW )  this->StoreSlidingMoves< SHIFT_SW >( mmap._SlidingMovesSW, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesSE )  this->StoreSlidingMoves< SHIFT_SE >( mmap._SlidingMovesSE, whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesN  )  this->StoreSlidingMoves< SHIFT_N  >( mmap._SlidingMovesN,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesW  )  this->StoreSlidingMoves< SHIFT_W  >( mmap._SlidingMovesW,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesE  )  this->StoreSlidingMoves< SHIFT_E  >( mmap._SlidingMovesE,  whitePieces, mmap._CheckMask );
        if( mmap._SlidingMovesS  )  this->StoreSlidingMoves< SHIFT_S  >( mmap._SlidingMovesS,  whitePieces, mmap._CheckMask );

        if( mmap._KnightMovesNNW )  this->StoreStepMoves( mmap._KnightMovesNNW, SHIFT_N + SHIFT_NW );
        if( mmap._KnightMovesNNE )  this->StoreStepMoves( mmap._KnightMovesNNE, SHIFT_N + SHIFT_NE );
        if( mmap._KnightMovesWNW )  this->StoreStepMoves( mmap._KnightMovesWNW, SHIFT_W + SHIFT_NW );
        if( mmap._KnightMovesENE )  this->StoreStepMoves( mmap._KnightMovesENE, SHIFT_E + SHIFT_NE );
        if( mmap._KnightMovesWSW )  this->StoreStepMoves( mmap._KnightMovesWSW, SHIFT_W + SHIFT_SW );
        if( mmap._KnightMovesESE )  this->StoreStepMoves( mmap._KnightMovesESE, SHIFT_E + SHIFT_SE );
        if( mmap._KnightMovesSSW )  this->StoreStepMoves( mmap._KnightMovesSSW, SHIFT_S + SHIFT_SW );
        if( mmap._KnightMovesSSE )  this->StoreStepMoves( mmap._KnightMovesSSE, SHIFT_S + SHIFT_SE );

        if( pos._BoardFlipped )
            this->FlipAll();
    }

    u64 FindMoves( const Position& pos )
    {
        MoveMap mmap;
        pos.CalcMoveMap( &mmap );

        this->UnpackMoveMap( pos, mmap );
        return( _Count );
    }

private:
    INLINE PDECL void StoreMove( int srcIdx, int destIdx, int promotion = 0 ) 
    {
        _Move[_Count++].Set( srcIdx, destIdx, promotion );
    }

    INLINE PDECL void StorePromotions( u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );

            this->StoreMove( idx - ofs, idx, PROMOTE_QUEEN );
            this->StoreMove( idx - ofs, idx, PROMOTE_ROOK );
            this->StoreMove( idx - ofs, idx, PROMOTE_BISHOP );
            this->StoreMove( idx - ofs, idx, PROMOTE_KNIGHT );
        }
    }

    INLINE PDECL void StoreStepMoves( u64 dest, int ofs ) 
    {
        while( dest )
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->StoreMove( idx - ofs, idx );
        }
    }

    template< int SHIFT >
    INLINE PDECL void StoreSlidingMoves( u64 dest, u64 pieces, u64 checkMask ) 
    {
        u64 src = Shift< -SHIFT >( dest ) & pieces;
        u64 cur = Shift<  SHIFT >( src );
        int ofs = SHIFT;

        while( cur )
        {
            this->StoreStepMoves( cur & checkMask, ofs );
            cur = Shift< SHIFT >( cur ) & dest;
            ofs += SHIFT;
        }
    }

    PDECL void StorePawnMoves( u64 dest, int ofs ) 
    {
        this->StoreStepMoves(  dest & ~RANK_8, ofs );
        this->StorePromotions( dest &  RANK_8, ofs );
    }

    PDECL void StoreKingMoves( u64 dest, u64 king ) 
    {
        int kingIdx = (int) LowestBitIndex( king );

        do
        {
            int idx = (int) ConsumeLowestBitIndex( dest );
            this->StoreMove( kingIdx, idx );
        }
        while( dest );
    }
};
