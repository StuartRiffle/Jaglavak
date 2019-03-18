// PlayoutCpu.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_CPU_H__
#define CORVID_PLAYOUT_CPU_H__


typedef std::shared_ptr< PlayoutJob >       PlayoutJobRef;
typedef ThreadSafeQueue< PlayoutJobRef >    PlayoutJobQueue;

typedef std::shared_ptr< PlayoutResult >    PlayoutResultRef;
typedef ThreadSafeQueue< PlayoutResultRef > PlayoutResultQueue;


extern CDECL ScoreCard PlayGamesSSE2(   const PlayoutJob& job, int simdCount )
extern CDECL ScoreCard PlayGamesSSE4(   const PlayoutJob& job, int simdCount )
extern CDECL ScoreCard PlayGamesAVX2(   const PlayoutJob& job, int simdCount )
extern CDECL ScoreCard PlayGamesAVX512( const PlayoutJob& job, int simdCount )


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

ScoreCard PlayGamesCpu( int cpuLevel, const PlayoutJob& job )
{
    int count       = job.mNumGames;
    int lanes       = PlatGetSimdWidth( cpuLevel );
    int simdCount   = (count + lanes - 1) / lanes;

    switch( cpuLevel )
    {
#if ENABLE_SSE2
    case CPU_SSE2:      
        return PlayGamesSSE2( job, simdCount );
#endif
#if ENABLE_SSE4
    case CPU_SSE4:      
        return PlayGamesSSE4( job, simdCount );
#endif
#if ENABLE_AVX2
    case CPU_AVX2:
        return PlayGamesAVX2( job, simdCount );
#endif
#if ENABLE_AVX512
    case CPU_AVX512:
        return PlayGamesAVX512( job, simdCount );
#endif
    }

    assert( cpuLevel == CPU_SCALAR );
    GamePlayer< u64 > scalarPlayer( &job.mOptions, mRandom.GetNext() );

    return scalarPlayer.PlayGames( job.mPosition, count );
}

PlayoutResult RunPlayoutJobCpu( const PlayoutJob& job )
{
    int cpuLevel = ChooseCpuLevelForPlayout( job.mOptions, job.mNumGames );

    JobResult result;
    result.mScores = PlayGamesCpu( cpuLevel, job );
    result.mPathFromRoot = job.mPathFromRoot;

    return result;
}


#endif
