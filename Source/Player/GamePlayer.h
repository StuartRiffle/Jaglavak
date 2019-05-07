// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "PlayoutBatch.h"

template< typename SIMD = u64 >
class GamePlayer
{
    const int LANES = SimdWidth< SIMD >::LANES;

    PlayoutParams*  mParams;
    RandomGen       mRandom;

public:

    PDECL GamePlayer( const PlayoutParams* params, int salt = 0 )
    {
        mParams = params;
        mRandom.SetSeed( params->mRandomSeed + salt );
    }

    PDECL void PlayGames( const PositionT< SIMD >* pos, ScoreCard* dest, int simdCount )
    {
        #pragma omp parallel for schedule(dynamic)
        for( int i = 0; i < simdCount; i++ )
        {
            PlayOneGameSimd( pos[i], dest + (i * LANES) );
        }
    }

protected:

    PDECL ScoreCard PlayOneGameSimd( const PositionT< SIMD >& startPos, ScoreCard* outScores )
    {
        PositionT< SIMD > simdPos = startPos;
        MoveMapT< SIMD > simdMoveMap;
        simdPos.CalcMoveMap( &simdMoveMap );

        for( int i = 0; i < mParams->mMaxMovesPerGame; i++ )
        {
            MoveSpecT< SIMD > simdSpec = this->ChoosePlayoutMovesSimd( simdPos, simdMoveMap );
            simdPos.Step( simdSpec, &simdMoveMap );
        }

        Position ALIGN_SIMD pos[LANES];
        Unswizzle< SIMD >( &simdPos, pos );

        for( int lane = 0; lane < LANES; lane++ )
        {
            outScores[lane].mWins[WHITE] += (pos[lane].mResult == RESULT_WHITE_WIN);
            outScores[lane].mWins[BLACK] += (pos[lane].mResult == RESULT_BLACK_WIN);
            outScores[lane].mPlays++;
        }
    }

    PDECL MoveSpecT< SIMD > ChoosePlayoutMovesSimd( const PositionT< SIMD >& simdPos, const MoveMapT< SIMD >& simdMoveMap )
    {
        // This part is still scalar :/

        Position ALIGN_SIMD pos[LANES];
        MoveMap  ALIGN_SIMD moveMap[LANES];
        MoveSpec ALIGN_SIMD spec[LANES];

        Unswizzle< SIMD >( &simdPos, pos );
        Unswizzle< SIMD >( &simdMoveMap, moveMap );

        for( int lane = 0; lane < LANES; lane++ )
            spec[lane] = this->ChoosePlayoutMove( pos[lane], moveMap[lane] );

        MoveSpecT< SIMD > simdSpec;
        simdSpec.Unpack( spec );

        return simdSpec;
    }

    PDECL MoveSpec ChoosePlayoutMove( const Position& pos, const MoveMap& moveMap )
    {
        if( pos.mResult == RESULT_UNKNOWN )
        {
            MoveList moveList;
            moveList.UnpackMoveMap( pos, moveMap );

            assert( moveList.mCount > 0 );
            int randomIdx = (int) mRandom.GetRange( moveList.mCount );

            return moveList.mMove[randomIdx];
        }

        MoveSpec nullMove( 0, 0, 0 );
        return nullMove;
    }
};

