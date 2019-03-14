// CPU-SSE2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_CPU_SSE2_H__
#define CORVID_CPU_SSE2_H__
#if ENABLE_SSE2

#include "Core.h"

CDECL ScoreCard PlayGamesSSE2( const PlayoutProvider* provider, const Position& pos, int simdCount )
{
    provider->PlayGamesSimd< simd2_sse2 >( pos, simdCount );
}

#endif // ENABLE_SSE2
#endif // CORVID_CPU_SSE2_H__
