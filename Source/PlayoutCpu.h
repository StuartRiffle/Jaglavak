// PlayoutCpu.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_CPU_H__
#define CORVID_PLAYOUT_CPU_H__


typedef std::shared_ptr< PlayoutJob >       PlayoutJobRef;
typedef ThreadSafeQueue< PlayoutJobRef >    PlayoutJobQueue;

typedef std::shared_ptr< PlayoutResult >    PlayoutResultRef;
typedef ThreadSafeQueue< PlayoutResultRef > PlayoutResultQueue;


int ChooseSimdLevelForPlayout( const GlobalOptions& options, int count )
{
    int simdLevel = 1;

#if SUPPORT_SSE4
    if( (count > 1) && (options.mDetectedSimdLevel >= 2) )
        simdLevel = 2;
#endif

#if SUPPORT_AVX2
    if( (count > 2) && (options.mDetectedSimdLevel >= 4) )
        simdLevel = 4;
#endif

#if SUPPORT_AVX512
    if( (count > 4) && (options.mDetectedSimdLevel >= 8) )
        simdLevel = 8;
#endif

    if( !options.mAllowSimd )
        simdLevel = CPU_SCALAR;

    if( options.mForceSimdLevel )
        simdLevel = options.mForceSimdLevel;

    return simdLevel;
}

template< typename SIMD >
ScoreCard PlayGamesCpu( const PlayoutJob& job, int simdCount )
{
    GamePlayer< SIMD > player( &job.mOptions, job.mRandomSeed );
    return player.PlayGames( job.mPosition, simdCount );
}

PlayoutResult RunPlayoutJobCpu( const PlayoutJob& job )
{
    int simdLevel   = ChooseSimdLevelForPlayout( job.mOptions, job.mNumGames );
    int simdCount   = (job.mNumGames + simdLevel - 1) / simdLevel;

    PlayoutResult result;
    result.mPathFromRoot = job.mPathFromRoot;

    switch( simdLevel )
    {
#if SUPPORT_SSE4
    case 2:
        result.mScores = PlayGamesCpu< simd2_sse4 >( job, simdCount );
        break;
#endif

#if SUPPORT_AVX2
    case 4:
        result.mScores = PlayGamesCpu< simd4_avx2 >( job, simdCount );
        break;
#endif

#if SUPPORT_AVX512
    case 8:
        result.mScores = PlayGamesCpu< simd8_avx512 >( job, simdCount );
        break;
#endif

    default:
        result.mScores = PlayGamesCpu< u64 >( job, simdCount );
        break;
    }

    return result;
}

#endif
