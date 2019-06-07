// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Random.h"
#include "PlayoutParams.h"

template< typename SIMD >
class GamePlayer
{
    enum { LANES = SimdWidth< SIMD >::LANES };

    const PlayoutParams* _Params;
    RandomGen _Random;

public:

    PDECL GamePlayer( const PlayoutParams* params, u64 salt = 0 )
    {
        _Params = params;
        _Random.SetSeed( params->_RandomSeed + salt );
    }

    PDECL void PlayGames( const Position* pos, ScoreCard* dest, int simdCount )
    {
        assert( (uintptr_t) pos % sizeof( SIMD ) == 0 );

        int totalIters = simdCount * _Params->_NumGamesEach;

        #pragma omp parallel for schedule(dynamic) if (_Params->_EnableMulticore)
        for( int i = 0; i < totalIters; i++ )
        {
            PositionT< SIMD > simdPos;
            int idx = i % simdCount;
            int offset = idx * LANES;

            Swizzle< SIMD >( pos + offset, &simdPos );
            PlayOneGame( simdPos, dest + offset );
        }
    }

protected:

    PDECL void PlayOneGame( const PositionT< SIMD >& startPos, ScoreCard* outScores )
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
        u64 rseed = _Random.s;

        MoveList moveList;
        MoveMap sparseMap;
        u64* buf = (u64*) &sparseMap;

        // All the fields in the MoveMap (up to _CheckMask) represent (potential) moves as bits

        const int count = (int) offsetof( MoveMap, _CheckMask ) / sizeof( u64 );
        sparseMap = moveMap;

        // Choose a lucky random bit (move) to keep

        u64 total = 0;
        for( int i = 0; i < count; i++ )
            total += PlatCountBits64( buf[i] );

        assert( total > 0 );
        u64 bitsToSkip = _Random.GetRange( total );

        // Find out which word it's in

        int word = 0;
        while( word < count )
        {
            u64 bits = PlatCountBits64( buf[word] );
            if( bits > bitsToSkip )
                break;

            bitsToSkip -= bits;
            buf[word++] = 0;
        }

        // Identify the bit and clear the rest

        u64 wordVal = buf[word];
        u64 destIdx = LowestBitIndex( wordVal );
        while( bitsToSkip-- )
        {
            wordVal = ClearLowestBit( wordVal );
            destIdx = LowestBitIndex( wordVal );
        }

        bool isSlidingMove = word < offsetof( MoveMap, _KnightMovesNNW );
        if( isSlidingMove )
        {
            assert( buf[word] & SquareBit( destIdx ) );
            int dir = word;

            static int shiftForDir[] =
            {
                SHIFT_N,
                SHIFT_NW,
                SHIFT_W,
                SHIFT_SW,
                SHIFT_S,
                SHIFT_SE,
                SHIFT_E,
                SHIFT_NE
            };

            u64 mask = SquareBit( destIdx );
            for( int i = 0; i < 7; i++ )
            {
                mask |= SignedShift( mask,  shiftForDir[dir] );
                mask |= SignedShift( mask, -shiftForDir[dir] );
            }

            assert( buf[dir] & mask );
            buf[dir] &= mask;
        }


        word++;
        while( word < count )
            buf[word++] = 0;

        moveList.UnpackMoveMap( pos, sparseMap );

        static u64 sCounts[MAX_POSSIBLE_MOVES] = { 0 };
        //sCounts[moveList._Count]++;

        for( int i = 0; i < moveList._Count; i++ )
        { 
            if( moveList._Move[i]._Dest == destIdx )
            {
                sCounts[moveList._Count]++;
                return moveList._Move[i];
            }
        }

        // The bit we chose is not a valid destination (maybe the move
        // would leave the king in check). So 

        if( moveList._Count == 0 )
            moveList.UnpackMoveMap( pos, moveMap );

        sCounts[moveList._Count]++;

        assert( moveList._Count > 0 );
        u64 idx = _Random.GetRange( moveList._Count );
        return moveList._Move[idx];
    }
};

