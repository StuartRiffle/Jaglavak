// CPU-AVX512.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_CPU_AVX512_H__
#define CORVID_CPU_AVX512_H__
#if ENABLE_AVX512

#include "Core.h"

CDECL ScoreCard PlayGamesAVX512( const PlayoutProvider* provider, const Position& pos, int simdCount )
{
    provider->PlayGamesSimd< simd8_avx512 >( pos, simdCount );
}

#endif // ENABLE_AVX512
#endif // CORVID_CPU_AVX512_H__
