// CPU-AVX2.cpp - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#include "Core.h"
#include "PlayoutJob.h"

#if ENABLE_AVX2
#include "SIMD-AVX2.h"

extern CDECL ScoreCard PlayGamesAVX2( const PlayoutJob& job, int simdCount )
{
    GamePlayer< simd4_avx2 > player( &job.mOptions, job.mRandomSeed );
    return( player.PlayGames( job.mPosition, simdCount ) );
}
#endif 
