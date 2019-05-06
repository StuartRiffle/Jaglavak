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
            PlayGameSimd( pos[i], dest + (i * LANES) );
        }
    }

protected:

    PDECL ScoreCard PlayGameSimd( const PositionT< SIMD >& startPos, ScoreCard* outScores )
    {
        Position ALIGN_SIMD pos[LANES];
        MoveMap  ALIGN_SIMD moveMap[LANES];
        MoveSpec ALIGN_SIMD spec[LANES];

        PositionT< SIMD > simdPos;
        MoveSpecT< SIMD > simdSpec;
        MoveMapT< SIMD >  simdMoveMap;

        simdPos = startPos;
        Unswizzle< SIMD >( &simdPos, pos );

        simdPos.CalcMoveMap( &simdMoveMap );
        Unswizzle< SIMD >( &simdMoveMap, moveMap );

        for( int i = 0; i < mParams->mMaxMovesPerGame; i++ )
        {
            for( int lane = 0; lane < LANES; lane++ )
                spec[lane] = this->ChoosePlayoutMove( pos[lane], moveMap[lane] );

            simdSpec.Unpack( spec );
            simdPos.Step( simdSpec, &simdMoveMap );
            
            Unswizzle< SIMD >( &simdPos, pos );
            Unswizzle< SIMD >( &simdMoveMap, moveMap );
        }

        for( int lane = 0; lane < LANES; lane++ )
        {
            outScores[lane].mWins[WHITE] += (pos[lane].mResult == RESULT_WHITE_WIN);
            outScores[lane].mWins[BLACK] += (pos[lane].mResult == RESULT_BLACK_WIN);
            outScores[lane].mPlays++;
        }
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

