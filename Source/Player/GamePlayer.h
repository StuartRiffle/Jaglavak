// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "PlayoutJob.h"

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

        for( int i = 0; i < mOptions->mPlayoutMaxMoves; i++ )
        {
            MoveSpecT< SIMD > simdSpec;
            MoveSpec spec[LANES];

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

                if( !laneDone[lane] )
                {
                    if( laneTargets[lane] == 0 )
                    {
                        laneFinalScore[lane] = score;
                        laneDone[lane] = true;
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

        return scores;
    }

    PDECL MoveSpec ChoosePlayoutMove( const Position& pos, const MoveMap& moveMap )
    {
        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );

        if( moveList.mCount == 0 )
        {
            MoveSpec nullMove( 0, 0, 0 );
            return nullMove;
        }

        int randomIdx = (int) mRandom.GetRange( moveList.mCount );
        return moveList.mMove[randomIdx];
    }
};

