// Playout.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_H__
#define CORVID_PLAYOUT_H__

struct ScoreCard
{
    u64 mWins;
    u64 mDraws;
    u64 mPlays;

    ScoreCard() : mWins( 0 ), mDraws( 0 ), mPlays( 0 ) {}

    ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins  += sc.mWins;
        mDraws += sc.mDraws;
        mPlays += sc.mPlays;

        return *this;
    }

    void FlipColor()
    {
        u64 losses = mPlays - (mWins + mDraws);
        mWins = losses;
    }
};

template< typename SIMD >
class GamePlayer
{
    const PlayoutOptions*   mOptions;
    RandomGen               mRandom;

public:

    GamePlayer( const PlayoutOptions* options ) : mOptions( options )
    {
        mRandom.SetSeed( options->mRandomSeed );
    }

    ScoreCard PlayGames( const Position& pos, int simdCount )
    {
    #if CORVID_CUDA_DEVICE

        ScoreCard scores;

        for( int i = 0; i < simdCount; i++ )
            scores += PlayGamesSimd< ENABLE_POPCNT >( pos );

        return scores;

    #else

        return mOptions->mUsePopcnt?
            PlayGamesThreaded< ENABLE_POPCNT  >( pos, simdCount ) :
            PlayGamesThreaded< DISABLE_POPCNT >( pos, simdCount );

    #endif
    }

protected:

    template< int POPCNT >
    ScoreCard PlayGamesThreaded( const Position& pos, int simdCount )
    {
        ScoreCard scores;

        #pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdScores = PlayGamesSimd< POPCNT >( pos );

            #pragma omp critical
            scores += simdScores;
        }

        return scores;
    }

    template< int POPCNT >
    ScoreCard PlayGamesSimd( const Position& startPos )
    {
        EvalWeightSet weights;

        float gamePhase = Evaluation::CalcGamePhase< POPCNT >( startPos );
        Evaluation::GenerateWeights( &weights, gamePhase );

        const int LANES = SimdWidth< SIMD >::LANES;

        Position ALIGN_SIMD pos[LANES];
        MoveMap  ALIGN_SIMD moveMap[LANES];

        PositionT< SIMD > simdPos;
        MoveMapT< SIMD > simdMoveMap;

        simdPos.Broadcast( startPos );
        simdPos.CalcMoveMap( &simdMoveMap );

        // This is the gameplay loop

        for( int i = 0; i < mOptions->mMaxPlayoutMoves; i++ )
        {
            MoveSpecT< SIMD > simdSpec;
            MoveSpec spec[LANES];

            for( int lane = 0; lane < LANES; lane++ )
                spec[lane] = this->ChoosePlayoutMove< POPCNT >( pos[lane], moveMap[lane], weights );

            Swizzle< SIMD >( pos, &simdPos );
            simdSpec.Unpack( spec );

            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );

            Unswizzle< SIMD >( &simdPos,     pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );
        }

        // Gather the results and judge them

        SIMD simdScore = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );
        u64* laneScore = (u64*) &simdScore;

        ScoreCard scores;

        for( int lane = 0; lane < LANES; lane++ )
        {
            bool whiteWon = (laneScore[lane] > mOptions->mAutoAdjudicate);
            bool blackWon = (laneScore[lane] < mOptions->mAutoAdjudicate);

            if( (whiteWon && pos[lane].mWhiteToMove) || (blackWon && !pos[lane].mWhiteToMove) )
                scores.mWins++;

            if( !whiteWon && !blackWon )
                scores.mDraws++;

            scores.mPlays++;
        }

        return scores;
    }

    template< int POPCNT >
    MoveSpec ChoosePlayoutMove( const Position& pos, const MoveMap& moveMap, const EvalWeightSet& weights )
    {
        int movesToPeek = mOptions->mMovesToPeek;
        bool makeErrorNow = (mRandom.GetFloat() < mOptions->mErrorRate);

        if( (movesToPeek < 1) || makeErrorNow )
        {
            // Fast path: completely random move

            return SelectRandomMove< POPCNT >( pos, moveMap );
        }

        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );

        if( moveList.mCount == 0 )
        {
            MoveSpec nullMove( 0, 0, 0 );
            return nullMove;
        }

        movesToPeek = Max( movesToPeek, moveList.mCount );

        // This has become a "heavy" playout, which means 
        // that we do static evaluation on a random subset 
        // of the moves

        MoveList peekList;
        EvalTerm eval[MAX_MOVE_LIST];

        while( movesToPeek > 0 )
        {
            const int LANES = SimdWidth< SIMD >::LANES;
            int numValid = Min( movesToPeek, LANES );

            // Pick some moves to evaluate

            MoveSpec spec[LANES];

            for( int i = 0; i < numValid; i++ )
            {
                int idx = (int) mRandom.GetRange( moveList.mCount );
                spec[i] = moveList.Remove( idx );
            }

            // Make those moves

            MoveSpecT< SIMD > simdSpec;
            PositionT< SIMD > simdPos;
            MoveMapT< SIMD >  simdMoveMap;

            simdSpec.Unpack( spec );
            simdPos.Broadcast( pos );
            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );

            // Evaluate the resulting positions

            SIMD simdScore = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );
            u64* laneScore = (u64*) &simdScore;

			for( int lane = 0; lane < numValid; lane++ )
            {
                eval[peekList.mCount] = (EvalTerm) laneScore[lane];
                peekList.Append( spec[lane] );
            }

            movesToPeek -= numValid;
        }

        // Find the highest score

        EvalTerm highestEval = eval[0];

        for( int i = 1; i < movesToPeek; i++ )
            if( eval[i] > highestEval )
                highestEval = eval[i];

        // Gather the moves with that score

        MoveList candidates;

        for( int i = 0; i < movesToPeek; i++ )
            if (eval[i] == highestEval )
                candidates.Append( peekList.mMove[i] );

        // Choose one of them at random

        int idx = (int) mRandom.GetRange( candidates.mCount );
        return moveList.mMove[idx];
    }

    template< int POPCNT >
    PDECL MoveSpec SelectRandomMove( const Position& pos, const MoveMap& moveMap )
    {
        MoveMap mmap = moveMap;

        // All the fields in the MoveMap (up to mCheckMask) represent moves as bits

        u64* buf = (u64*) &mmap;
        u64 count = (u64) (((u64*) mmap.mCheckMask) - buf);

        // Choose a random bit to keep

        u64 total = 0;
        for( int i = 0; i < count; i++ )
            total += CountBits< POPCNT >( buf[i] );

        u64 bitsToSkip = mRandom.GetRange( total );

        // Find which word it's in

        int word = 0;
        while( word < count )
        {
            u64 bits = CountBits< POPCNT >( buf[word] );
            if( bits >= bitsToSkip )
                break;

            bitsToSkip -= bits;
            buf[word] = 0;
            word++;
        }

        // Keep just that one, and clear the rest

        u64 idx;
        while( bitsToSkip-- )
            idx = ConsumeLowestBitIndex( buf[word] );

        buf[word] = 1ULL << idx;
        word++;

        while( word < count )
            buf[word++] = 0;

        // That should give us exactly one move, unless it's a pawn being
        // promoted, in which case there will be four, but we'll just
        // use first one anyway (queen).

        MoveList moveList;
        moveList.UnpackMoveMap( pos, mmap );

        assert( (moveList.mCount == 1) || (moveList.mCount == 4) );

        return moveList.mMove[0];
    }
};

#if !CORVID_CUDA_DEVICE

extern CDECL ScoreCard PlayGamesSSE2(   const PlayoutOptions& options, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesSSE4(   const PlayoutOptions& options, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesAVX2(   const PlayoutOptions& options, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesAVX512( const PlayoutOptions& options, const Position& pos, int simdCount );

static ScoreCard PlayGamesCpu( const PlayoutOptions& options, const Position& pos, int count )
{
    int cpuLevel =
        (count > 4)? CPU_AVX512 :
        (count > 2)? CPU_AVX2 :
        (count > 1)? CPU_SSE4 : 
                     CPU_SCALAR;

    if( cpuLevel > options.mMaxCpuLevel )
        cpuLevel = options.mMaxCpuLevel;

    if( options.mForceCpuLevel != CPU_INVALID )
        cpuLevel = options.mForceCpuLevel;

    int lanes = PlatGetSimdWidth( cpuLevel );
    int simdCount = (count + lanes - 1) / lanes;

    switch( cpuLevel )
    {
#if ENABLE_SSE2
    case CPU_SSE2:   
        return PlayGamesSSE2( options, pos, simdCount );
#endif
#if ENABLE_SSE4
    case CPU_SSE4:
        return PlayGamesSSE4( options, pos, simdCount );
#endif
#if ENABLE_AVX2
    case CPU_AVX2:
        return PlayGamesAVX2( options, pos, simdCount );
#endif
#if ENABLE_AVX512
    case CPU_AVX512: 
        return PlayGamesAVX512( options, pos, simdCount );
#endif
    }

    // Fall back to scalar

    GamePlayer< u64 > player( &options );
    return( player.PlayGames( pos, count ) );
}

#endif // !CORVID_CUDA_DEVICE


#endif // CORVID_PLAYOUT_H__
