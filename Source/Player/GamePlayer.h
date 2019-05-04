// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "PlayoutJob.h"

template< typename SIMD >
class GamePlayer
{
    RandomGen   mRandom;
    int         mMaxMoves;

public:

    PDECL GamePlayer( u64 randomSeed, int maxMoves )
    {
        mRandom.SetSeed( randomSeed );
        mMaxMoves = maxMoves;
    }

    PDECL ScoreCard PlayGames( const Position& pos, int simdCount )
    {
        ScoreCard scores;

        #pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            ScoreCard simdScores = PlayGamesSimd( pos );

            #pragma omp critical
            scores += simdScores;
        }

        return scores;
    }

protected:

    PDECL ScoreCard PlayGamesSimd( const PositionT< SIMD >& simdPos )
    {
        const int LANES = SimdWidth< SIMD >::LANES;

        Position ALIGN_SIMD pos[LANES];
        MoveMap  ALIGN_SIMD moveMap[LANES];

        MoveMapT< SIMD > simdMoveMap;
        simdPos.CalcMoveMap( &simdMoveMap );

        Unswizzle< SIMD >( &simdPos,     pos );
        Unswizzle< SIMD >( &simdMoveMap, moveMap );

        // This is the gameplay loop

        int laneResult[LANES] = { 0 };

        for( int i = 0; i < mMaxMoves; i++ )
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

            int lanesDone = 0;
            for( int lane = 0; lane < LANES; lane++ )
            {
                if( !laneResult[lane] )
                    laneResult[lane] = pos.CalcGameResult( pos[lane], moveMap[lane] );

                if( laneResult[lane] )
                    lanesDone++;
            }

            if( lanesDone == LANES )
                break;
        }

        // Compile the results

        ScoreCard scores;

        for( int lane = 0; lane < LANES; lane++ )
        {
            if( laneResult[lane] == RESULT_WHITE_WIN )
                scores.mWins[WHITE]++;

            if( laneResult[lanel] == RESULT_BLACK_WIN )
                scores.mWins[BLACK]++

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

