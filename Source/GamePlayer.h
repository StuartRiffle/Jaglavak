// Playout.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_H__
#define CORVID_PLAYOUT_H__

struct ScoreCard
{
    u64 mWins;
    u64 mDraws;
    u64 mPlays;

    void Clear()
    {
        mWins = mDraws = mPlays = 0;
    }

    void FlipColor()
    {
        u64 losses = mPlays - (mWins + mDraws);
        mWins = losses;
    }

    ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins  += sc.mWins;
        mDraws += sc.mDraws;
        mPlays += sc.mPlays;
        return *this;
    }

};

template< typename SIMD >
class GamePlayer
{
    const GlobalOptions*    mOptions;
    RandomGen               mRandom;

public:

    GamePlayer( const GlobalOptions* options, u64 randomSeed ) : mOptions( options )
    {
        mRandom.SetSeed( randomSeed );
    }

    ScoreCard PlayGames( const Position& pos, int simdCount )
    {
        return mOptions->mEnablePopcnt?
            PlayGamesThreaded< ENABLE_POPCNT  >( pos, simdCount ) :
            PlayGamesThreaded< DISABLE_POPCNT >( pos, simdCount );
    }

protected:

    template< int POPCNT >
    ScoreCard PlayGamesThreaded( const Position& pos, int simdCount )
    {
        PROFILER_SCOPE( "GamePlayer::PlayGamesThreaded" );

        ScoreCard scores;
        scores.Clear();

        #pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdScores = PlayGamesSimd< POPCNT >( pos );

            #pragma omp critical
            scores.mWins  += simdScores.mWins;
            scores.mDraws += simdScores.mDraws;
            scores.mPlays += simdScores.mPlays;
        }

        return scores;
    }

    template< int POPCNT >
    ScoreCard PlayGamesSimd( const Position& startPos )
    {
        PROFILER_SCOPE( "GamePlayer::PlayGamesSimd" );

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

        for( int i = 0; i < mOptions->mPlayoutMaxMoves; i++ )
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
        scores.Clear();

        for( int lane = 0; lane < LANES; lane++ )
        {
            bool whiteWon = (laneScore[lane] >  mOptions->mWinningMaterial);
            bool blackWon = (laneScore[lane] < -mOptions->mWinningMaterial);

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
        PROFILER_SCOPE( "GamePlayer::ChoosePlayoutMove" );

        int movesToPeek = mOptions->mPlayoutPeekMoves;
        bool makeErrorNow = (mRandom.GetRange( 100 ) < mOptions->mPlayoutErrorRate);

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

            SIMD simdScore;
            simdScore = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );

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
        PROFILER_SCOPE( "GamePlayer::SelectRandomMove" );

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


#endif // CORVID_PLAYOUT_H__
