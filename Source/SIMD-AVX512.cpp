// CPU-AVX512.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "PlayoutJob.h"

#if ENABLE_AVX512
#include "SIMD-AVX512.h"

CDECL ScoreCard PlayGamesAVX512( const PlayoutJob& job, int simdCount )
{
    GamePlayer< simd8_avx512 > player( &job.mOptions, job.mRandomSeed );
    return( player.PlayGames( job.mPosition, simdCount ) );
}
#endif 
