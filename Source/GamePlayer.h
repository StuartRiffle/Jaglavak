// Playout.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_H__
#define CORVID_PLAYOUT_H__

struct ScoreCard
{
    u64 mWins;
    u64 mDraws;
    u64 mPlays;

    PDECL ScoreCard() : mWins( 0 ), mDraws( 0 ), mPlays( 0 ) {}

    PDECL void FlipColor()
    {
        u64 losses = mPlays - (mWins + mDraws);
        mWins = losses;
    }

    PDECL ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins  += sc.mWins;
        mDraws += sc.mDraws;
        mPlays += sc.mPlays;
        return *this;
    }

    PDECL void Print( const char* desc )
    {
        DEBUG_LOG( "%s scores %d %d %d\n", desc, mWins, mDraws, mPlays );
    }
};

template< typename SIMD >
class GamePlayer
{
    const GlobalOptions*    mOptions;
    RandomGen               mRandom;

public:

    PDECL GamePlayer( const GlobalOptions* options, u64 randomSeed ) : mOptions( options )
    {
        mRandom.SetSeed( randomSeed );
    }

    PDECL ScoreCard PlayGames( const Position& pos, int simdCount )
    {
        return mOptions->mEnablePopcnt?
            PlayGamesThreaded< ENABLE_POPCNT  >( pos, simdCount ) :
            PlayGamesThreaded< DISABLE_POPCNT >( pos, simdCount );
    }

protected:

    template< int POPCNT >
    PDECL ScoreCard PlayGamesThreaded( const Position& pos, int simdCount )
    {
        PROFILER_SCOPE( "GamePlayer::PlayGamesThreaded" );

        ScoreCard scores;

        //#pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdScores = PlayGamesSimd< POPCNT >( pos );

            //#pragma omp critical
            scores += simdScores;
        }

        return scores;
    }

    template< int POPCNT >
    PDECL ScoreCard PlayGamesSimd( const Position& startPos )
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

            Unswizzle< SIMD >( &simdPos,     pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );

            // FIXME Unswizzle SSE4 corruption

            // FIXME pos/movemap mismatch?
            
            for( int lane = 0; lane < LANES; lane++ )
                spec[lane] = this->ChoosePlayoutMove< POPCNT >( pos[lane], moveMap[lane], weights );

            Swizzle< SIMD >( pos, &simdPos );
            simdSpec.Unpack( spec );

            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );
        }

        // Gather the results and judge them

        SIMD simdScore = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );
        u64* laneScores = (u64*) &simdScore;

        ScoreCard scores;

        for( int lane = 0; lane < LANES; lane++ )
        {
            EvalTerm laneScore = (EvalTerm) laneScores[lane];

            bool whiteWon = (laneScore >  mOptions->mWinningMaterial);
            bool blackWon = (laneScore < -mOptions->mWinningMaterial);

            if( (whiteWon && pos[lane].mWhiteToMove) || (blackWon && !pos[lane].mWhiteToMove) )
                scores.mWins++;

            if( !whiteWon && !blackWon )
                scores.mDraws++;

            DEBUG_LOG( "Lane %d, final eval %d, whiteWon %d, blackWon %d\n", lane, laneScore, whiteWon? 1 : 0, blackWon? 1 : 0 );

            scores.mPlays++;
        }

        scores.Print( "PlayGamesSimd" );
        return scores;
    }

    template< int POPCNT >
    PDECL MoveSpec ChoosePlayoutMove( const Position& pos, const MoveMap& moveMap, const EvalWeightSet& weights )
    {
        PROFILER_SCOPE( "GamePlayer::ChoosePlayoutMove" );

        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );

        if( moveList.mCount == 0 )
        {
            DEBUG_LOG( "NULL move!" );

            MoveSpec nullMove( 0, 0, 0 );
            return nullMove;
        }

        int movesToPeek = mOptions->mPlayoutPeekMoves;
        bool makeErrorNow = (mRandom.GetRange( 100 ) < mOptions->mPlayoutErrorRate);

        if( (movesToPeek < 1) || makeErrorNow )
        {
            // Fast path: completely random move

            int randomIdx = (int) mRandom.GetRange( moveList.mCount );

            //std::string spec = SerializeMoveSpec( moveList.mMove[randomIdx] );
            //DEBUG_LOG( "Choosing move %d (%s)\n", randomIdx, spec.c_str() );

            return moveList.mMove[randomIdx];
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

        MoveMap moveMapCopy = moveMap;

        // All the fields in the MoveMap (up to mCheckMask) represent moves as bits

        u64* buf = (u64*) &moveMapCopy;
        u64 count = (u64) (((u64*) &moveMapCopy.mCheckMask) - buf);

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
        moveList.UnpackMoveMap( pos, moveMapCopy );

        if( moveList.mCount == 0 )
        {
            MoveMap debugMoveMap;
            pos.CalcMoveMap( &debugMoveMap );

            moveList.UnpackMoveMap( pos, moveMapCopy );
            moveList.UnpackMoveMap( pos, moveMap );
        }
        else
        {
            assert( (moveList.mCount == 1) || (moveList.mCount == 4) );
        }

        return moveList.mMove[0];
    }
};


#endif // CORVID_PLAYOUT_H__
