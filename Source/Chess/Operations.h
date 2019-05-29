// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
struct SimdWidth
{                                                       
    enum { LANES = 1 };
};

// These are the functions that SIMD types need to implement

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
template< typename T > INLINE PDECL T       Min( const T& a, const T& b )                                       { return( (a < b)? a : b ); }
template< typename T > INLINE PDECL T       Max( const T& a, const T& b )                                       { return( (b > a)? b : a ); }
template< typename T > INLINE PDECL T       SignOrZero( const T& val )                                          { return( (val > 0) - (val < 0) ); }
template< typename T > INLINE PDECL T       SquareBit( const T& idx )                                           { return( T( 1 ) << idx ); }
template< typename T > INLINE PDECL T       LowestBit( const T& val )                                           { return( val & -val ); }
template< typename T > INLINE PDECL T       ClearLowestBit( const T& val )                                      { return( val & (val - 1) ); }
template< typename T > INLINE PDECL T       FlipSquareIndex( const T& idx )                                     { return( ((T( 63 ) - idx) & 0x38) | (idx & 0x7) ); }
template< typename T > INLINE PDECL T       ClearBitIndex( const T& val, const T& idx )                         { return( val & ~SquareBit( idx ) ); }
template< typename T > INLINE PDECL T       LowestBitIndex( const T& val )                                      { return PlatLowestBitIndex64( val ); }
template< typename T > INLINE PDECL T       ConsumeLowestBitIndex( T& val )                                     { T idx = LowestBitIndex( val ); val = ClearLowestBit( val ); return( idx ); }
template< typename T > INLINE PDECL void    Exchange( T& a, T& b )                                              { T temp = a; a = b; b = temp; }
template< typename T > INLINE PDECL void    Transpose( const T* src, int srcStep, T* dest, int destStep )       { *dest = *src; }

template< typename SIMD, typename UNPACKED, typename PACKED >
INLINE PDECL void Swizzle( const UNPACKED* srcStruct, PACKED* destStruct )
{
    const int LANES = SimdWidth< SIMD >::LANES;

    int blockSize  = sizeof( SIMD ) * LANES;
    int blockCount = (int) sizeof( PACKED ) / blockSize;
    int srcStride  = sizeof( UNPACKED ) / sizeof( SIMD );

    const SIMD* RESTRICT    src     = (SIMD*) srcStruct;
    SIMD* RESTRICT          dest    = (SIMD*) destStruct;

    while( blockCount-- )
    {
        Transpose< SIMD >( src, srcStride, dest, 1 );

        src  += 1;
        dest += LANES;
    }
}

template< typename SIMD, typename PACKED, typename UNPACKED >
INLINE PDECL void Unswizzle( const PACKED* srcStruct, UNPACKED* destStruct )
{
    const int LANES = SimdWidth< SIMD >::LANES;

    int blockSize  = sizeof( SIMD ) * LANES;
    int blockCount = (int) sizeof( PACKED ) / blockSize;
    int destStride = sizeof( UNPACKED ) / sizeof( SIMD );

    const SIMD* RESTRICT    src     = (SIMD*) srcStruct;
    SIMD* RESTRICT          dest    = (SIMD*) destStruct;

    while( blockCount-- )
    {
        Transpose< SIMD >( src, 1, dest, destStride );

        src  += LANES;
        dest += 1;
    }
}

