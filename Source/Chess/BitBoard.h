// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#define ANY (ALL_SQUARES)

template< int SHIFT, typename T >
INLINE PDECL T Shift( T bits )
{
    if( SHIFT > 0 )
        return( bits << (SHIFT) );
    else 
        return( bits >> ((-SHIFT)) );
}

template< int SHIFT, typename T >
INLINE PDECL T Propagate( T bits, T allow )
{
    T v     = bits;
    T mask  = allow;

    v    |= Shift< SHIFT     >( v ) & mask;
    mask &= Shift< SHIFT     >( mask );
    v    |= Shift< SHIFT * 2 >( v ) & mask;
    mask &= Shift< SHIFT * 2 >( mask );
    v    |= Shift< SHIFT * 4 >( v ) & mask;

    return( v );
}

template< typename T >
INLINE PDECL T StepKnights( T n, T allow = ANY )
{
    //  . C . C .
    //  D . . . D
    //  b a(n)a b
    //  D . . . D
    //  . C . C .

    T a = Shift< SHIFT_W >( n & ~FILE_A ) | Shift< SHIFT_E >( n & ~FILE_H );                                  
    T b = Shift< SHIFT_W * 2 >( n & ~(FILE_A | FILE_B) ) | Shift< SHIFT_E * 2 >( n & ~(FILE_G | FILE_H) );    
    T c = Shift< SHIFT_N * 2 >( a ) | Shift< SHIFT_S * 2 >( a );                                                  
    T d = Shift< SHIFT_N >( b ) | Shift< SHIFT_S >( b );                                                          

    return( (c | d) & allow );                                                                                    
}

template< typename T > INLINE PDECL T   StepN(         T val, T allow = ANY )   { return( Shift< SHIFT_N  >( val ) & allow ); }
template< typename T > INLINE PDECL T   StepNW(        T val, T allow = ANY )   { return( Shift< SHIFT_NW >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL T   StepW(         T val, T allow = ANY )   { return( Shift< SHIFT_W  >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL T   StepSW(        T val, T allow = ANY )   { return( Shift< SHIFT_SW >( val ) & allow & ~FILE_H ); }
template< typename T > INLINE PDECL T   StepS(         T val, T allow = ANY )   { return( Shift< SHIFT_S  >( val ) & allow ); }
template< typename T > INLINE PDECL T   StepSE(        T val, T allow = ANY )   { return( Shift< SHIFT_SE >( val ) & allow & ~FILE_A ); }
template< typename T > INLINE PDECL T   StepE(         T val, T allow = ANY )   { return( Shift< SHIFT_E  >( val ) & allow & ~FILE_A ); }
template< typename T > INLINE PDECL T   StepNE(        T val, T allow = ANY )   { return( Shift< SHIFT_NE >( val ) & allow & ~FILE_A ); }

template< typename T > INLINE PDECL T   StepOrtho(     T val, T allow = ANY )   { return( StepN(     val, allow ) | StepW(    val, allow ) | StepS(  val, allow ) | StepE ( val, allow ) ); }
template< typename T > INLINE PDECL T   StepDiag(      T val, T allow = ANY )   { return( StepNW(    val, allow ) | StepSW(   val, allow ) | StepSE( val, allow ) | StepNE( val, allow ) ); }
template< typename T > INLINE PDECL T   StepOut(       T val, T allow = ANY )   { return( StepOrtho( val, allow ) | StepDiag( val, allow ) ); }

template< typename T > INLINE PDECL T   PropN(         T val, T allow = ANY )   { return( Propagate< SHIFT_N  >( val, allow ) ); }
template< typename T > INLINE PDECL T   PropNW(        T val, T allow = ANY )   { return( Propagate< SHIFT_NW >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL T   PropW(         T val, T allow = ANY )   { return( Propagate< SHIFT_W  >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL T   PropSW(        T val, T allow = ANY )   { return( Propagate< SHIFT_SW >( val, allow & ~FILE_H ) ); }
template< typename T > INLINE PDECL T   PropS(         T val, T allow = ANY )   { return( Propagate< SHIFT_S  >( val, allow ) ); }
template< typename T > INLINE PDECL T   PropSE(        T val, T allow = ANY )   { return( Propagate< SHIFT_SE >( val, allow & ~FILE_A ) ); }
template< typename T > INLINE PDECL T   PropE(         T val, T allow = ANY )   { return( Propagate< SHIFT_E  >( val, allow & ~FILE_A ) ); }
template< typename T > INLINE PDECL T   PropNE(        T val, T allow = ANY )   { return( Propagate< SHIFT_NE >( val, allow & ~FILE_A ) ); }

template< typename T > INLINE PDECL T   PropOrtho(     T val, T allow = ANY )   { return( PropN(  val, allow ) | PropW(  val, allow ) | PropS(  val, allow ) | PropE(  val, allow ) ); }
template< typename T > INLINE PDECL T   PropDiag(      T val, T allow = ANY )   { return( PropNW( val, allow ) | PropSW( val, allow ) | PropSE( val, allow ) | PropNE( val, allow ) ); }

template< typename T > INLINE PDECL T   PropExN(       T val, T allow = ANY )   { return( MaskOut( PropN(  val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExNW(      T val, T allow = ANY )   { return( MaskOut( PropNW( val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExW(       T val, T allow = ANY )   { return( MaskOut( PropW(  val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExSW(      T val, T allow = ANY )   { return( MaskOut( PropSW( val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExS(       T val, T allow = ANY )   { return( MaskOut( PropS(  val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExSE(      T val, T allow = ANY )   { return( MaskOut( PropSE( val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExE(       T val, T allow = ANY )   { return( MaskOut( PropE(  val, allow ), val ) ); }
template< typename T > INLINE PDECL T   PropExNE(      T val, T allow = ANY )   { return( MaskOut( PropNE( val, allow ), val ) ); }

template< typename T > INLINE PDECL T   PropExOrtho(   T val, T allow = ANY )   { return( PropExN(  val, allow ) | PropExW(  val, allow ) | PropExS(  val, allow ) | PropExE(  val, allow ) ); }
template< typename T > INLINE PDECL T   PropExDiag(    T val, T allow = ANY )   { return( PropExNW( val, allow ) | PropExSW( val, allow ) | PropExSE( val, allow ) | PropExNE( val, allow ) ); }

template< typename T > INLINE PDECL T   SlideIntoN(    T val, T through, T into )       { T acc = PropN(  val, through ); acc |= StepN(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoNW(   T val, T through, T into )       { T acc = PropNW( val, through ); acc |= StepNW( acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoW(    T val, T through, T into )       { T acc = PropW(  val, through ); acc |= StepW(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoSW(   T val, T through, T into )       { T acc = PropSW( val, through ); acc |= StepSW( acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoS(    T val, T through, T into )       { T acc = PropS(  val, through ); acc |= StepS(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoSE(   T val, T through, T into )       { T acc = PropSE( val, through ); acc |= StepSE( acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoE(    T val, T through, T into )       { T acc = PropE(  val, through ); acc |= StepE(  acc, into ); return( acc ); }
template< typename T > INLINE PDECL T   SlideIntoNE(   T val, T through, T into )       { T acc = PropNE( val, through ); acc |= StepNE( acc, into ); return( acc ); }

template< typename T > INLINE PDECL T   SlideIntoExN(  T val, T through, T into )       { T acc = PropN(  val, through ); T poke = StepN(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExNW( T val, T through, T into )       { T acc = PropNW( val, through ); T poke = StepNW( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExW(  T val, T through, T into )       { T acc = PropW(  val, through ); T poke = StepW(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExSW( T val, T through, T into )       { T acc = PropSW( val, through ); T poke = StepSW( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExS(  T val, T through, T into )       { T acc = PropS(  val, through ); T poke = StepS(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExSE( T val, T through, T into )       { T acc = PropSE( val, through ); T poke = StepSE( acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExE(  T val, T through, T into )       { T acc = PropE(  val, through ); T poke = StepE(  acc, into ); return( MaskOut( acc, val ) | poke ); }
template< typename T > INLINE PDECL T   SlideIntoExNE( T val, T through, T into )       { T acc = PropNE( val, through ); T poke = StepNE( acc, into ); return( MaskOut( acc, val ) | poke ); }

template< typename T > INLINE PDECL T   SlideIntoExOrtho( T val, T through, T into )    { return( SlideIntoExN(  val, through, into ) | SlideIntoExW(  val, through, into ) | SlideIntoExS(  val, through, into ) | SlideIntoExE(  val, through, into ) ); }
template< typename T > INLINE PDECL T   SlideIntoExDiag(  T val, T through, T into )    { return( SlideIntoExNW( val, through, into ) | SlideIntoExSW( val, through, into ) | SlideIntoExSE( val, through, into ) | SlideIntoExNE( val, through, into ) ); }
