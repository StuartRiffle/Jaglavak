// Playout.h - CORVID CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef CORVID_PLAYOUT_H__
#define CORVID_PLAYOUT_H__

struct ScoreCard
{
    u64 mWins[2];
    u64 mPlays;

    PDECL ScoreCard()
    {
        mWins[0] = 0;
        mWins[1] = 0;
        mPlays = 0;
    }

    PDECL ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins[0] += sc.mWins[0];
        mWins[1] += sc.mWins[1];
        mPlays += sc.mPlays;
        return *this;
    }

    PDECL void Print( const char* desc )
    {
        //DEBUG_LOG( "%s scores %d %d %d\n", desc, mWins, mDraws, mPlays );
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

        Unswizzle< SIMD >( &simdPos,     pos );
        Unswizzle< SIMD >( &simdMoveMap, moveMap );

        // This is the gameplay loop

        bool laneDone[LANES] = { false };
        EvalTerm laneFinalScore[LANES];

        for( int i = 0; i < mOptions->mPlayoutMaxMoves; i++ )
        {
            MoveSpecT< SIMD > simdSpec;
            MoveSpec spec[LANES];

            // FIXME Unswizzle SSE4 corruption

            // FIXME pos/movemap mismatch?

            // FIXME moveMap[lane] i
            
            for( int lane = 0; lane < LANES; lane++ )
                spec[lane] = this->ChoosePlayoutMove< POPCNT >( pos[lane], moveMap[lane], weights );

            Swizzle< SIMD >( pos, &simdPos );
            simdSpec.Unpack( spec );

            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );

            Unswizzle< SIMD >( &simdPos,     pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );

            // Detect games that are done

            SIMD simdScores = Evaluation::EvaluatePosition< POPCNT, SIMD >( simdPos, simdMoveMap, weights );
            u64* laneScores = (u64*) &simdScores;

            SIMD simdTargets = simdMoveMap.CalcMoveTargets();
            u64* laneTargets = (u64*) &simdTargets;

            SIMD simdInCheck = simdMoveMap.IsInCheck();
            u64* laneInCheck = (u64*) &simdInCheck;

            int numDone = 0;

            // FIXME: evaluation is always from white POV?

            for( int lane = 0; lane < LANES; lane++ )
            {
                if( !laneDone[lane] )
                {
                    if( laneTargets[lane] == 0 )
                    {
                        laneFinalScore[lane] = (EvalTerm) laneScores[lane];
                        laneDone[lane] = true;
                    }
                }
            
                if( laneDone[lane] )
                    numDone++;
            }

            if( numDone == LANES )
                break;
        }

        // Gather the results and judge them

        ScoreCard scores;

        for( int lane = 0; lane < LANES; lane++ )
        {
            if( laneDone[lane] )
            {
                bool whiteWon = (laneFinalScore[lane] >  mOptions->mWinningMaterial);
                bool blackWon = (laneFinalScore[lane] < -mOptions->mWinningMaterial);

                if( blackWon && !startPos.mWhiteToMove )
                    scores.mWins[0]++;

                if( whiteWon && startPos.mWhiteToMove )
                    scores.mWins[1]++;
            }

            scores.mPlays++;
        }

        //scores.Print( "PlayGamesSimd" );
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
            //DEBUG_LOG( "NULL move!\n" );

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

        movesToPeek = Min( movesToPeek, moveList.mCount );

        // This has become a "heavy" playout, which means that
        // we do static evaluation on a subset of the moves.

        // TODO: prefer non-quiet moves
        //MoveList specialMoves;
        //moveList.CopySpecialMoves( &specialMoves );

        MoveList peekList;
        EvalTerm eval[MAX_MOVE_LIST];

        while( movesToPeek > 0 )
        {
            const int LANES = SimdWidth< SIMD >::LANES;

            int numValid = Min( moveList.mCount, Min( movesToPeek, LANES ) );
            if( numValid == 0 )
                break;

            // Pick some moves to evaluate

            MoveSpec spec[LANES];

            for( int i = 0; i < numValid; i++ )
            {
                int idx = (int) mRandom.GetRange( moveList.mCount );

                for( int j = 0; j < moveList.mCount; j++ )
                    if( moveList.mMove[j].IsCapture() )
                        idx = j;

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

        for( int i = 1; i < peekList.mCount; i++ )
            if( eval[i] > highestEval )
                highestEval = eval[i];

        // Gather the moves with that score

        MoveList candidates;

        for( int i = 0; i < peekList.mCount; i++ )
            if (eval[i] == highestEval)
                candidates.Append( peekList.mMove[i] );

        // Choose one of them at random

        int idx = (int) mRandom.GetRange( candidates.mCount );
        return moveList.mMove[idx];
    }
};


#endif // CORVID_PLAYOUT_H__
