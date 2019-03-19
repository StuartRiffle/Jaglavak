// PlayoutCpu.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_CPU_H__
#define CORVID_PLAYOUT_CPU_H__


typedef std::shared_ptr< PlayoutJob >       PlayoutJobRef;
typedef ThreadSafeQueue< PlayoutJobRef >    PlayoutJobQueue;

typedef std::shared_ptr< PlayoutResult >    PlayoutResultRef;
typedef ThreadSafeQueue< PlayoutResultRef > PlayoutResultQueue;


int ChooseCpuLevelForPlayout( const GlobalOptions& options, int count )
{
    int cpuLevel = CPU_SCALAR;

#if ENABLE_SSE2
    if( (count > 1) && (options.mDetectedCpuLevel >= CPU_SSE2) )
        cpuLevel = CPU_SSE2;
#endif
#if ENABLE_SSE4
    if( (count > 1) && (options.mDetectedCpuLevel >= CPU_SSE4) )
        cpuLevel = CPU_SSE4;
#endif
#if ENABLE_AVX2
    if( (count > 2) && (options.mDetectedCpuLevel >= CPU_AVX2) )
        cpuLevel = CPU_AVX2;
#endif
#if ENABLE_AVX512
    if( (count > 4) && (options.mDetectedCpuLevel >= CPU_AVX512) )
        cpuLevel = CPU_AVX512;
#endif

    if( !options.mEnableSimd )
        cpuLevel = CPU_SCALAR;

    if( options.mForceCpuLevel != CPU_INVALID )
        cpuLevel = options.mForceCpuLevel;

    return cpuLevel;
}

template< typename SIMD >
ScoreCard PlayGamesCpu( const PlayoutJob& job, int simdCount )
{
    GamePlayer< SIMD > player( &job.mOptions, job.mRandomSeed );
    return player.PlayGames( job.mPosition, simdCount );
}

PlayoutResult RunPlayoutJobCpu( const PlayoutJob& job )
{
    int cpuLevel    = ChooseCpuLevelForPlayout( job.mOptions, job.mNumGames );
    int lanes       = PlatGetSimdWidth( cpuLevel );
    int simdCount   = (job.mNumGames + lanes - 1) / lanes;

    PlayoutResult result;
    result.mPathFromRoot = job.mPathFromRoot;

    switch( cpuLevel )
    {
#if ENABLE_SSE2
    case CPU_SSE2: 
        result.mScores = PlayGamesCpu< simd2_sse2 >( job, simdCount );
        break;
#endif
#if ENABLE_SSE4
    case CPU_SSE4:
        result.mScores = PlayGamesCpu< simd2_sse4 >( job, simdCount );
        break;
#endif
#if ENABLE_AVX2
    case CPU_AVX2:
        result.mScores = PlayGamesCpu< simd4_avx2 >( job, simdCount );
        break;
#endif
#if ENABLE_AVX512
    case CPU_AVX512:
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
