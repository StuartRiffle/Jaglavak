// CPU-AVX2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_CPU_AVX2_H__
#define CORVID_CPU_AVX2_H__
#if ENABLE_AVX2

#include "Core.h"

CDECL ScoreCard PlayGamesAVX2( const PlayoutProvider* provider, const Position& pos, int simdCount )
{
    provider->PlayGamesSimd< simd4_avx2 >( pos, simdCount );
}

#endif // ENABLE_AVX2
#endif // CORVID_CPU_AVX2_H__
