// Operations.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#pragma once

template< typename T >
struct SimdWidth
{                                                       
    enum { LANES = 1 };
};

template< typename T > INLINE PDECL T       MaskAllClear()                                                      { return(  T( 0 ) ); }
template< typename T > INLINE PDECL T       MaskAllSet()                                                        { return( ~T( 0 ) ); }
template< typename T > INLINE PDECL T       MaskOut( const T& val, const T& bitsToClear )                       { return( val & ~bitsToClear ); }
template< typename T > INLINE PDECL T       SelectIfNotZero( const T& val, const T& a )                         { return( val? a : 0 ); }
template< typename T > INLINE PDECL T       SelectIfNotZero( const T& val, const T& a, const T& b )             { return( val? a : b ); }
template< typename T > INLINE PDECL T       SelectIfZero(    const T& val, const T& a )                         { return( val? 0 : a ); }
template< typename T > INLINE PDECL T       SelectIfZero(    const T& val, const T& a, const T& b )             { return( val? b : a ); }
template< typename T > INLINE PDECL T       SelectWithMask(  const T& mask, const T& a, const T& b )            { return( b ^ (mask & (a ^ b)) ); } 
template< typename T > INLINE PDECL T       CmpEqual( const T& a, const T& b )                                  { return( (a == b)? MaskAllSet< T >() : MaskAllClear< T >() ); }
template< typename T > INLINE PDECL T       ByteSwap( const T& val )                                            { return PlatByteSwap64( val ); }
template< typename T > INLINE PDECL T       MulSigned32( const T& val, i32 scale )                              { return( val * scale ); }
template< typename T > INLINE PDECL T       SubClampZero( const T& a, const T& b )                              { return( (a > b)? (a - b) : 0 ); }
template< typename T > INLINE PDECL T       Min( const T& a, const T& b )                                       { return( (a < b)? a : b ); }
template< typename T > INLINE PDECL T       Max( const T& a, const T& b )                                       { return( (b > a)? b : a ); }
template< typename T > INLINE PDECL T       SignOrZero( const T& val )                                          { return( (val > 0) - (val < 0) ); }
template< typename T > INLINE PDECL T       SquareBit( const T& idx )                                           { return( T( 1 ) << idx ); }
template< typename T > INLINE PDECL T       LowestBit( const T& val )                                           { return( val & -val ); }
template< typename T > INLINE PDECL T       ClearLowestBit( const T& val )                                      { return( val & (val - 1) ); }
template< typename T > INLINE PDECL T       FlipSquareIndex( const T& idx )                                     { return( ((T( 63 ) - idx) & 0x38) | (idx & 0x7) ); }
template< typename T > INLINE PDECL T       XorShiftA( const T& val )                                           { T n = val; return( n ^= (n << 18), n ^= (n >> 31), n ^= (n << 11), n ); }    
template< typename T > INLINE PDECL T       XorShiftB( const T& val )                                           { T n = val; return( n ^= (n << 19), n ^= (n >> 29), n ^= (n <<  8), n ); }    
template< typename T > INLINE PDECL T       XorShiftC( const T& val )                                           { T n = val; return( n ^= (n <<  8), n ^= (n >> 29), n ^= (n << 19), n ); }    
template< typename T > INLINE PDECL T       XorShiftD( const T& val )                                           { T n = val; return( n ^= (n << 11), n ^= (n >> 31), n ^= (n << 18), n ); }    
template< typename T > INLINE PDECL T       ClearBitIndex( const T& val, const T& idx )                         { return( val & ~SquareBit( idx ) ); }
template< typename T > INLINE PDECL T       LowestBitIndex( const T& val )                                      { return PlatLowestBitIndex64( val ); }
template< typename T > INLINE PDECL T       ConsumeLowestBitIndex( T& val )                                     { T idx = LowestBitIndex( val ); val = ClearLowestBit( val ); return( idx ); }
template< typename T > INLINE PDECL void    Exchange( T& a, T& b )                                              { T temp = a; a = b; b = temp; }
template< typename T > INLINE PDECL void    Transpose( const T* src, int srcStep, T* dest, int destStep )       { *dest = *src; }
template< typename T >        PDECL void    SimdInsert( T& dest, u64 val, int lane )                            { dest = val; }
