// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

template< typename T >
struct SimdWidth
{                                                       
    enum { LANES = 1 };
};

// These are the functions that SIMD types need to implement

template< typename T > INLINE PDECL T    MaskAllClear()                      { return(  T( 0 ) ); }
template< typename T > INLINE PDECL T    MaskAllSet()                        { return( ~T( 0 ) ); }
template< typename T > INLINE PDECL T    MaskOut( T val, T bitsToClear )     { return( val & ~bitsToClear ); }
template< typename T > INLINE PDECL T    SelectIfNotZero( T val, T a )       { return( val? a : 0 ); }
template< typename T > INLINE PDECL T    SelectIfNotZero( T val, T a, T b )  { return( val? a : b ); }
template< typename T > INLINE PDECL T    SelectIfZero( T val, T a )          { return( val? 0 : a ); }
template< typename T > INLINE PDECL T    SelectIfZero( T val, T a, T b )     { return( val? b : a ); }
template< typename T > INLINE PDECL T    SelectWithMask(  T mask, T a, T b ) { return( b ^ (mask & (a ^ b)) ); } 
template< typename T > INLINE PDECL T    CmpEqual( T a, T b )                { return( (a == b)? MaskAllSet< T >() : MaskAllClear< T >() ); }
template< typename T > INLINE PDECL T    ByteSwap( T val )                   { return PlatByteSwap64( val ); }
template< typename T > INLINE PDECL T    Min( T a, T b )                     { return( (a < b)? a : b ); }
template< typename T > INLINE PDECL T    Max( T a, T b )                     { return( (b > a)? b : a ); }
template< typename T > INLINE PDECL T    SignOrZero( T val )                 { return( (val > 0) - (val < 0) ); }
template< typename T > INLINE PDECL T    SquareBit( T idx )                  { return( T( 1 ) << idx ); }
template< typename T > INLINE PDECL T    LowestBit( T val )                  { return( val & -val ); }
template< typename T > INLINE PDECL T    ClearLowestBit( T val )             { return( val & (val - 1) ); }
template< typename T > INLINE PDECL T    FlipSquareIndex( T idx )            { return( ((T( 63 ) - idx) & 0x38) | (idx & 0x7) ); }
template< typename T > INLINE PDECL T    ClearBitIndex( T val, T idx )       { return( val & ~SquareBit( idx ) ); }
template< typename T > INLINE PDECL T    LowestBitIndex( T val )             { return PlatLowestBitIndex64( val ); }
template< typename T > INLINE PDECL T    ConsumeLowestBitIndex( T& val )     { T idx = LowestBitIndex( val ); val = ClearLowestBit( val ); return( idx ); }
template< typename T > INLINE PDECL void Exchange( T& a, T& b )              { T temp = a; a = b; b = temp; }
template< typename T > INLINE PDECL void Transpose( const T* src, int srcStep, T* dest, int destStep ) { *dest = *src; }

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

