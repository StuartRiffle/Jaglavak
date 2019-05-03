// Playout.h - JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#ifndef JAGLAVAK_PLAYOUT_H__
#define JAGLAVAK_PLAYOUT_H__

struct ScoreCard
{
    u64 mWins[2];
    u64 mPlays;

    PDECL ScoreCard()
    {
        this->Clear();
    }

    PDECL void Clear()
    {
        mWins[BLACK] = 0;
        mWins[WHITE] = 0;
        mPlays = 0;
    }

    PDECL ScoreCard& operator+=( const ScoreCard& sc )
    {
        mWins[BLACK] += sc.mWins[BLACK];
        mWins[WHITE] += sc.mWins[WHITE];
        mPlays += sc.mPlays;
        return *this;
    }

    PDECL void Print( const char* desc )
    {
        //DEBUG_LOG( "%s scores %d %d plays %d\n", desc, mWins[WHITE], mWins[BLACK], mPlays );
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
        PROFILER_SCOPE( "GamePlayer::PlayGames" );

        ScoreCard scores;

        #pragma omp parallel for schedule(dynamic) if (mOptions->mAllowParallel)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdScores = PlayGamesSimd( pos );

            #pragma omp critical
            scores += simdScores;
        }

        return scores;
    }

protected:

    PDECL ScoreCard PlayGamesSimd( const Position& startPos )
    {
        PROFILER_SCOPE( "GamePlayer::PlayGamesSimd" );

        EvalWeightSet weights;

        float gamePhase = Evaluation::CalcGamePhase( startPos );
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
                spec[lane] = this->ChoosePlayoutMove( pos[lane], moveMap[lane], weights );

            Swizzle< SIMD >( pos, &simdPos );
            simdSpec.Unpack( spec );

            simdPos.Step( simdSpec );
            simdPos.CalcMoveMap( &simdMoveMap );

            Unswizzle< SIMD >( &simdPos,     pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );

            // Detect games that are done

            SIMD simdScores = Evaluation::EvaluatePosition< SIMD >( simdPos, simdMoveMap, weights );
            u64* laneScores = (u64*) &simdScores;

            SIMD simdTargets = simdMoveMap.CalcMoveTargets();
            u64* laneTargets = (u64*) &simdTargets;

            SIMD simdInCheck = simdMoveMap.IsInCheck();
            u64* laneInCheck = (u64*) &simdInCheck;

            int numDone = 0;

            for( int lane = 0; lane < LANES; lane++ )
            {
                EvalTerm score = (EvalTerm) laneScores[lane];

                // Score is always from white's POV

                if( !pos[lane].mWhiteToMove )
                    score = -score;

                //DEBUG_LOG( "Iter %d score %d fen %s\n", i, score, SerializePosition( pos[lane] ).c_str() ); 

                if( !laneDone[lane] )
                {
                    if( laneTargets[lane] == 0 )
                    {
                        laneFinalScore[lane] = score;
                        laneDone[lane] = true;

                        //DEBUG_LOG("FINAL RESULT %d IS %d fen %s\n", i, laneFinalScore[lane], SerializePosition( pos[lane] ).c_str());
                    }

                    u64 nonKingPieces =
                        pos[lane].mWhitePawns |  
                        pos[lane].mWhiteKnights |
                        pos[lane].mWhiteBishops |
                        pos[lane].mWhiteRooks |
                        pos[lane].mWhiteQueens |  
                        pos[lane].mBlackPawns |  
                        pos[lane].mBlackKnights |
                        pos[lane].mBlackBishops |
                        pos[lane].mBlackRooks |
                        pos[lane].mBlackQueens; 

                    if( nonKingPieces == 0 )
                    {
                        laneFinalScore[lane] = 0;
                        laneDone[lane] = true;

                        //DEBUG_LOG("DRAW BY INSUFFICIENT MATERIAL %d fen %s\n", i, SerializePosition( pos[lane] ).c_str() );
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
                bool whiteWon = (laneFinalScore[lane] > 0);
                bool blackWon = (laneFinalScore[lane] < 0);

                if( whiteWon )
                    scores.mWins[WHITE]++;

                if( blackWon )
                    scores.mWins[BLACK]++;
            }

            scores.mPlays++;
        }

        //scores.Print( "PlayGamesSimd" );
        return scores;
    }

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
            simdScore = Evaluation::EvaluatePosition< SIMD >( simdPos, simdMoveMap, weights );

            u64* laneScore = (u64*) &simdScore;

			for( int lane = 0; lane < numValid; lane++ )
            {
                EvalTerm score = (EvalTerm) laneScore[lane];


                // Moves abs high are best for white
                // Moves 


                // FIXME
                if( !pos.mWhiteToMove )
                    score = -score;

                eval[peekList.mCount] = (EvalTerm) laneScore[lane];
                peekList.Append( spec[lane] );
            }

            movesToPeek -= numValid;
        }


        // FIXME: the sign of the score is flipping, so the "highest" is sometimes lowest


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


#endif // JAGLAVAK_PLAYOUT_H__
