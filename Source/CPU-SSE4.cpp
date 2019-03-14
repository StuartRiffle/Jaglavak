// CPU-SSE4.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_CPU_SSE4_H__
#define CORVID_CPU_SSE4_H__
#if ENABLE_SSE4

#include "Core.h"

CDECL ScoreCard PlayGamesSSE4( const PlayoutProvider* provider, const Position& pos, int simdCount )
{
    provider->PlayGamesSimd< simd2_sse4 >( pos, simdCount );
}

#endif // ENABLE_SSE4
#endif // CORVID_CPU_SSE4_H__
