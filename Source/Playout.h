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
};

struct PlayoutProvider;
extern CDECL ScoreCard PlayGamesSSE2(   const PlayoutProvider* provider, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesSSE4(   const PlayoutProvider* provider, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesAVX2(   const PlayoutProvider* provider, const Position& pos, int simdCount );
extern CDECL ScoreCard PlayGamesAVX512( const PlayoutProvider* provider, const Position& pos, int simdCount );

struct PlayoutOptions
{
    int     mErrorRate;
    int     mMovesToPeek;
    int     mMaxPlayoutMoves;
    int     mAutoAdjudicate;
    u64     mRandomSeed;

    int     mMaxCpuLevel;
    int     mForceCpuLevel;
    bool    mUsePopcnt;
};


struct PlayoutJobInfo
{
    Position            mPosition;
    PlayoutOptions      mOptions;
    int                 mNumGames;
    MoveList            mPathFromRoot;
};

struct PlayoutJobResult
{
    ScoreCard           mScores;
    MoveList            mPathFromRoot;
};






struct PlayoutProvider
{
    PlayoutOptions* mOptions;
    RandomGen       mRandom;

    PlayoutProvider( PlayoutOptions* options ) : mOptions( options )
    {
        mRandom.SetSeed( options->mRandomSeed );
    }

    template< int POPCNT >
    PDECL MoveSpec SelectRandomMove( const Position& pos, const MoveMap& moveMap )
    {
        MoveMap mmap = moveMap;

        // Only look at fields before mCheckMask

        u64* buf = (u64*) &mmap;
        int count = (int) (((u64*) mmap.mCheckMask) - buf);

        // Choose a random bit to keep

        int total = 0;
        for( int i = 0; i < count; i++ )
            total += (int) CountBits< POPCNT >( buf[i] );

        int bitsToSkip = (int) mRandom.GetRange( total );

        int word = 0;
        while( word < count )
        {
            int bits = (int) CountBits< POPCNT >( buf[word] );
            if( bits >= bitsToSkip )
                break;

            bitsToSkip -= bits;
            buf[word] = 0;
            word++;
        }

        u64 idx;
        while( bitsToSkip-- )
            idx = ConsumeLowestBitIndex( buf[word] );

        buf[word] = 1ULL << idx;
        word++;

        while( word < count )
            buf[word++] = 0;

        MoveList moveList;
        moveList.UnpackMoveMap( pos, mmap );

        assert( (moveList.mCount == 1) || (moveList.mCount == 4) );

        return moveList.mMove[0];
    }

    template< int POPCNT, typename SIMD >
    MoveSpec ChoosePlayoutMove( const Position& pos, const MoveMap& moveMap, const EvalWeightSet& weights )
    {
        int movesToPeek = mOptions->mMovesToPeek;
        bool makeError = (mRandom.GetFloat() < mOptions->mErrorRate);

        if( (movesToPeek < 1) || makeError )
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

    template< int POPCNT, typename SIMD >
    ScoreCard PlayGames( const Position& startPos )
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
                spec[lane] = this->ChoosePlayoutMove< POPCNT, SIMD >( pos[lane], moveMap[lane], weights );

            Swizzle< SIMD >( pos, &simdPos );
            simdSpec.Unpack( spec );

            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );

            Unswizzle< SIMD >( &simdPos,     pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );
        }

        // Gather the results and judge them

        SIMD simdScore = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );
        u64* laneScore = (u64*) simdScore;

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

    template< int POPCNT, typename SIMD >
    ScoreCard PlayGamesThreaded( const Position& pos, int simdCount )
    {
        ScoreCard scores;

        #pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdResults = PlayGames< POPCNT, SIMD >( pos );

            #pragma omp critical
            scores += simdResults;
        }

        return scores;
    }

    template< typename SIMD >
    ScoreCard PlayGamesSimd( const Position& pos, int simdCount )
    {
        return mOptions->mUsePopcnt?
            PlayGamesThreaded< ENABLE_POPCNT,  SIMD >( pos, simdCount ) :
            PlayGamesThreaded< DISABLE_POPCNT, SIMD >( pos, simdCount );
    }

    ScoreCard DoPlayouts( const Position& pos, int count )
    {
        int cpuLevel =
            (count > 4)? CPU_AVX512 :
            (count > 2)? CPU_AVX2 :
            (count > 1)? CPU_SSE4 : 
                         CPU_X64;

        if( cpuLevel > mOptions->mMaxCpuLevel )
            cpuLevel = mOptions->mMaxCpuLevel;

        if( mOptions->mForceCpuLevel != CPU_INVALID )
            cpuLevel = mOptions->mForceCpuLevel;

        int lanes = PlatGetSimdWidth( cpuLevel );
        int simdCount = (count + lanes - 1) / lanes;

        switch( cpuLevel )
        {
        case CPU_SSE2:   return PlayGamesSSE2(   this, pos, simdCount );
        case CPU_SSE4:   return PlayGamesSSE4(   this, pos, simdCount );
        case CPU_AVX2:   return PlayGamesAVX2(   this, pos, simdCount );
        case CPU_AVX512: return PlayGamesAVX512( this, pos, simdCount );
        default:         return PlayGamesSimd< u64 >(  pos, simdCount );
        }
    }
};

#endif // CORVID_PLAYOUT_H__
