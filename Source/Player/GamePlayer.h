// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Random.h"
#include "PlayoutParams.h"

template< typename SIMD >
class GamePlayer
{
    enum { LANES = SimdWidth< SIMD >::LANES };

    const PlayoutParams* _Params;
    RandomGen _RandomGen;

public:

    PDECL GamePlayer( const PlayoutParams* params, u64 salt = 0 )
    {
        _Params = params;
        _RandomGen.SetSeed( params->_RandomSeed + salt );
    }

    PDECL void PlayGames( const Position* pos, ScoreCard* dest, int simdCount )
    {
        assert( (uintptr_t) pos % sizeof( SIMD ) == 0 );

        #if !ON_CUDA_DEVICE
            int totalCores = PlatDetectCpuCores();
            int coresToUse = _Params->_LimitPlayoutCores;
            if( coresToUse < 0 )
                coresToUse += totalCores;
            omp_set_num_threads( coresToUse );
        #endif

        int totalIters = simdCount * _Params->_NumGamesEach;

        #pragma omp parallel for schedule(dynamic) if(totalIters)
        for( int i = 0; i < totalIters; i++ )
        {
        #if !ON_CUDA_DEVICE
            PlatLimitCores( coresToUse, false );
        #endif

            PositionT< SIMD > simdPos;
            int idx = i % simdCount;
            int offset = idx * LANES;

            Swizzle< SIMD >( pos + offset, &simdPos );
            ___PLAYOUT___( simdPos, dest + offset );
        }
    }

protected:

    PDECL void ___PLAYOUT___( const PositionT< SIMD >& startPos, ScoreCard* outScores )
    {
        PositionT< SIMD > simdPos = startPos;
        MoveMapT< SIMD >  simdMoveMap;
        simdPos.CalcMoveMap( &simdMoveMap );

        for( int i = 0; i < _Params->_MaxMovesPerGame; i++ )
        {
            MoveSpecT< SIMD > simdSpec = ChoosePlayoutMoves( simdPos, simdMoveMap );

            simdPos.Step( simdSpec, &simdMoveMap );
            if( GamesAreAllDone( simdPos ) )
                break;
        }

        u64* results = (u64*) &simdPos._GameResult;
        for( int lane = 0; lane < LANES; lane++ )
        {
            outScores[lane]._Wins[WHITE] += (results[lane] == RESULT_WHITE_WIN);
            outScores[lane]._Wins[BLACK] += (results[lane] == RESULT_BLACK_WIN);
            outScores[lane]._Plays++;
        }
    }

    PDECL MoveSpecT< SIMD > ChoosePlayoutMoves( const PositionT< SIMD >& simdPos, const MoveMapT< SIMD >& simdMoveMap )
    {
        Position ALIGN_SIMD pos[LANES];
        MoveMap  ALIGN_SIMD moveMap[LANES];
        MoveSpec ALIGN_SIMD spec[LANES];

        Unswizzle< SIMD >( &simdPos, pos );
        Unswizzle< SIMD >( &simdMoveMap, moveMap );

        for( int lane = 0; lane < LANES; lane++ )
            spec[lane] = this->ChooseMove( pos[lane], moveMap[lane] );

        MoveSpecT< SIMD > simdSpec;
        simdSpec.Unpack( spec );

        return simdSpec;
    }

    PDECL MoveSpec ChooseMove( const Position& pos, const MoveMap& moveMap )
    {
        if( pos._GameResult == RESULT_UNKNOWN )
        {
            MoveSpec randomMove = SelectRandomMove( pos, moveMap );
            return randomMove;
        }

        MoveSpec nullMove( 0, 0, 0 );
        return nullMove;
    }

    PDECL bool GamesAreAllDone( const PositionT< SIMD >& simdPos )
    {
        u64* results = (u64*) &simdPos._GameResult;

        for( int i = 0; i < LANES; i++ )
            if( results[i] == RESULT_UNKNOWN )
                return false;

        return true;
    }

    PDECL MoveSpec SelectRandomMove( const Position& pos, const MoveMap& moveMap )
    {
        MoveList moveList;
        moveList.UnpackMoveMap( pos, moveMap );
        assert( moveList._Count > 0 );

        u64 idx = _RandomGen.GetRange( moveList._Count );
        return moveList._Move[idx];
    }
};

