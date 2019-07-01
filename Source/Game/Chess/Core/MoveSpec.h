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
    INLINE PDECL MoveSpecT( T src, T dest, T type = 0 ) : _Src( src ), _Dest( dest ), _Type( type ) {}
    INLINE PDECL void Set( T src, T dest, T type = 0 ) { _Src = src;  _Dest = dest;   _Type = type; }

    template< typename U > INLINE PDECL MoveSpecT( const MoveSpecT< U >& rhs ) : _Src( rhs._Src ), _Dest( rhs._Dest ), _Type( rhs._Type ) {}

    INLINE PDECL int  IsCapture() const { return(((_Type >= CAPTURE_LOSING) && (_Type <= CAPTURE_WINNING)) || ((_Type >= CAPTURE_PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN))); }
    INLINE PDECL int  IsPromotion() const { return((_Type >= PROMOTE_KNIGHT) && (_Type <= CAPTURE_PROMOTE_QUEEN)); }
    INLINE PDECL void Flip() { _Src = FlipSquareIndex( _Src ); _Dest = FlipSquareIndex( _Dest ); }
    INLINE PDECL char GetPromoteChar() const { return("\0\0\0\0nbrqnbrq\0\0"[_Type]); }

    INLINE PDECL bool operator==( const MoveSpecT& rhs ) const { return((_Src == rhs._Src) && (_Dest == rhs._Dest) && (_Type == rhs._Type)); }
    INLINE PDECL bool operator!=( const MoveSpecT& rhs ) const { return((_Src != rhs._Src) || (_Dest != rhs._Dest) || (_Type != rhs._Type)); }

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
        return(memcmp( this, &other, sizeof( *this ) ) < 0);
    }
};
