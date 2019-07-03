// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "catch.hpp"

template< typename T >
struct TestOperations
{
    void Run()
    {
        SCENARIO( "Branch-free operations" )
        {
            int LANES = ( int) T::LANES;

            REQUIRE( T( -1 ) == T( ~0 ) );
            REQUIRE( sizeof( T ) % 8 == 0 );
            REQUIRE( LANES & ~(LANES - 1) == 0 );

            REQUIRE( MaskAllClear() == T( 0 ) );
            REQUIRE( MaskAllSet() == T( ~0 ) );

            REQUIRE( MaskOut( T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( ~0 ), T( 0 ) ) == T( 0 ) );

            REQUIRE( MaskOut( T( 0 ), T( ~0 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( 1 ), T( ~0 ) ) == T( 1 ) );
            REQUIRE( MaskOut( T( ~0 ), T( ~0 ) ) == T( ~0 ) );

            REQUIRE( MaskOut( T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( MaskOut( T( ~0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( MaskOut( T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( 1 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( MaskOut( T( 2 ), T( 1 ) ) == T( 2 ) );
            REQUIRE( MaskOut( T( 3 ), T( 1 ) ) == T( 2 ) );

            REQUIRE( SelectIfNotZero( T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( SelectIfNotZero( T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 2 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectIfNotZero( T( 0 ), T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfNotZero( T( 0 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 0 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfNotZero( T( 0 ), T( 0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectIfNotZero( T( 1 ), T( 1 ), T( 0 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 1 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 1 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfNotZero( T( 1 ), T( 0 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( SelectIfNotZero( T( 2 ), T( 1 ), T( 0 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 2 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfNotZero( T( 2 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfNotZero( T( 2 ), T( 0 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( SelectIfZero( T( 0 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfZero( T( 1 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 2 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( SelectIfZero( T( 0 ), T( 1 ), T( 0 ) ) == T( 1 ) );
            REQUIRE( SelectIfZero( T( 0 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfZero( T( 0 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 0 ), T( 0 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( SelectIfZero( T( 1 ), T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 1 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfZero( T( 1 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 1 ), T( 0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectIfZero( T( 2 ), T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 2 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectIfZero( T( 2 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectIfZero( T( 2 ), T( 0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectWithMask( T( 0 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 0 ), T( 0 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectWithMask( T( 0 ), T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 0 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectWithMask( T( 0 ), T( 2 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 0 ), T( 2 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectWithMask( T( 1 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 1 ), T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 1 ), T( 1 ), T( 0 ) ) == T( 1 ) );
            REQUIRE( SelectWithMask( T( 1 ), T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( SelectWithMask( T( 1 ), T( 2 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 1 ), T( 2 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SelectWithMask( T( 2 ), T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 2 ), T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 2 ), T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 2 ), T( 1 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( SelectWithMask( T( 2 ), T( 2 ), T( 0 ) ) == T( 2 ) );
            REQUIRE( SelectWithMask( T( 2 ), T( 2 ), T( 1 ) ) == T( 2 ) );

            REQUIRE( CmpEqual( T( 0 ), T( 0 ) ) == T( ~0 ) );
            REQUIRE( CmpEqual( T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( CmpEqual( T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( CmpEqual( T( 1 ), T( 1 ) ) == T( ~0 ) );
            REQUIRE( CmpEqual( T( 2 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( CmpEqual( T( 2 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( ByteSwap( T( 0x12345678 ) ) == T( 0x87654321 ) );
            REQUIRE( ByteSwap( T( 0x87654321 ) ) == T( 0x12345678 ) );

            REQUIRE( Min( T( ~0 ), T( ~0 ) ) == T( ~0 ) );
            REQUIRE( Min( T( ~0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( Min( T( ~0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( Min( T( 0 ), T( ~0 ) ) == T( 0 ) );
            REQUIRE( Min( T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( Min( T( 0 ), T( 1 ) ) == T( 0 ) );

            REQUIRE( Min( T( 1 ), T( ~0 ) ) == T( 1 ) );
            REQUIRE( Min( T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( Min( T( 1 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( Max( T( ~0 ), T( ~0 ) ) == T( ~0 ) );
            REQUIRE( Max( T( ~0 ), T( 0 ) ) == T( ~0 ) );
            REQUIRE( Max( T( ~0 ), T( 1 ) ) == T( ~0 ) );

            REQUIRE( Max( T( 0 ), T( ~0 ) ) == T( ~0 ) );
            REQUIRE( Max( T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( Max( T( 0 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( Max( T( 1 ), T( ~0 ) ) == T( ~0 ) );
            REQUIRE( Max( T( 1 ), T( 0 ) ) == T( 1 ) );
            REQUIRE( Max( T( 1 ), T( 1 ) ) == T( 1 ) );

            REQUIRE( SignOrZero( T( 2 ) ) == T( 1 ) );
            REQUIRE( SignOrZero( T( 1 ) ) == T( 1 ) );
            REQUIRE( SignOrZero( T( 0 ) ) == T( 0 ) );
            REQUIRE( SignOrZero( T( -1 ) ) == T( -1 ) );
            REQUIRE( SignOrZero( T( -2 ) ) == T( -1 ) );

            REQUIRE( SquareBit( T( 64 ) ) == T( 0 ) );
            for( u64 i = 0; i < 64; i <<= 1 )
                REQUIRE( SquareBit( T( i ) ) == T( 1ULL << i ) );

            REQUIRE( LowestBit( T( ~0 ) ) == T( 0 ) );
            REQUIRE( LowestBit( T( 0 ) ) == T( 0 ) );
            REQUIRE( LowestBit( T( 1 ) ) == T( 1 ) );
            REQUIRE( LowestBit( T( 2 ) ) == T( 2 ) );
            REQUIRE( LowestBit( T( 3 ) ) == T( 1 ) );
            REQUIRE( LowestBit( T( 4 ) ) == T( 4 ) );

            REQUIRE( ClearLowestBit( T( 0 ) ) == T( 0 ) );
            REQUIRE( ClearLowestBit( T( 1 ) ) == T( 0 ) );
            REQUIRE( ClearLowestBit( T( 2 ) ) == T( 0 ) );
            REQUIRE( ClearLowestBit( T( 3 ) ) == T( 2 ) );
            REQUIRE( ClearLowestBit( T( 4 ) ) == T( 0 ) );
            REQUIRE( ClearLowestBit( T( 5 ) ) == T( 4 ) );
            REQUIRE( ClearLowestBit( T( 6 ) ) == T( 4 ) );
            REQUIRE( ClearLowestBit( T( 7 ) ) == T( 6 ) );
            REQUIRE( ClearLowestBit( T( 8 ) ) == T( 0 ) );

            for( int x = 0; x < 8; x++ )
                for( int y = 0; y < 8; y++ )
                    REQUIRE( FlipSquareIndex( (x << 4) | y ) == FlipSquareIndex( (y << 4) | x );)

                    REQUIRE( ClearBitIndex( T( 0 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 0 ), T( 1 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 0 ), T( 2 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 0 ), T( 3 ) ) == T( 0 ) );

            REQUIRE( ClearBitIndex( T( 1 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 1 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( ClearBitIndex( T( 1 ), T( 2 ) ) == T( 2 ) );
            REQUIRE( ClearBitIndex( T( 1 ), T( 3 ) ) == T( 2 ) );

            REQUIRE( ClearBitIndex( T( 2 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 2 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( ClearBitIndex( T( 2 ), T( 2 ) ) == T( 2 ) );
            REQUIRE( ClearBitIndex( T( 2 ), T( 3 ) ) == T( 1 ) );

            REQUIRE( ClearBitIndex( T( 3 ), T( 0 ) ) == T( 0 ) );
            REQUIRE( ClearBitIndex( T( 3 ), T( 1 ) ) == T( 1 ) );
            REQUIRE( ClearBitIndex( T( 3 ), T( 2 ) ) == T( 2 ) );
            REQUIRE( ClearBitIndex( T( 3 ), T( 3 ) ) == T( 3 ) );

            REQUIRE( LowestBitIndex( T( 0 ) ) == T( ~0 ) );
            REQUIRE( LowestBitIndex( T( 1 ) ) == T( 0 ) );
            REQUIRE( LowestBitIndex( T( 2 ) ) == T( 1 ) );
            REQUIRE( LowestBitIndex( T( 3 ) ) == T( 0 ) );
            REQUIRE( LowestBitIndex( T( 4 ) ) == T( 2 ) );
            REQUIRE( LowestBitIndex( T( 5 ) ) == T( 0 ) );
            REQUIRE( LowestBitIndex( T( 6 ) ) == T( 2 ) );
            REQUIRE( LowestBitIndex( T( 7 ) ) == T( 0 ) );
            REQUIRE( LowestBitIndex( T( 8 ) ) == T( 3 ) );

            T val = ~0;
            for( int i = 0; i < 64; i++ )
                REQUIRE( ConsumeLowestBitIndex( val ) == i );

            T a( 0 );
            T b( ~0 );
            Exchange( a, b );
            REQUIRE( (a == T( ~0 )) && (b == T( 0 )) );
            Exchange( a, b );
            REQUIRE( (a == T( 0 )) && (b == T( ~0 )) );

            u64 SIMD_ALIGN src[LANES * LANES * 2];
            u64 SIMD_ALIGN dest[LANES * LANES * 2];

            for( int srcStep = 0; srcStep < 2; srcStep++ )
            {
                for( int destStep = 0; destStep < 2; destStep++ )
                {
                    memset( src, 0, sizeof( src ) );
                    memset( dest, 0, sizeof( dest ) );

                    for( int y = 0; y < LANES; y++ )
                        for( int x = 0; x < LANES; x++ )
                            src[y * lanes * srcStep + x * destStep] = (y * lanes * destStep) + x * srcStep;

                    Transpose( src, srcStep, dest, destStep );

                    for( int y = 0; y < LANES; y++ )
                        for( int x = 0; x < LANES; x++ )
                            REQUIRE( dest[y * lanes * srcStep + x * destStep] == (x * lanes * destStep) + y * srcStep );
                }
            }
        }
    }
};

