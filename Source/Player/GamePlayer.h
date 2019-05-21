






// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "Random.h"
#include "PlayoutParams.h"

#if !ON_CUDA_DEVICE
#include <stdio.h>
static void PositionToString2( const Position& p, char* str )
{
    Position pos = p;
    if( pos.mBoardFlipped )
    {
        Position flipped;
        flipped.FlipFrom( pos );
    }

    char* cur = str;

    for( u64 bit = 1ULL << 63; bit != 0; bit >>= 1 )
    {
        if( pos.mWhitePawns   & bit )   *cur++ = 'P'; else
        if( pos.mWhiteKnights & bit )   *cur++ = 'N'; else
        if( pos.mWhiteBishops & bit )   *cur++ = 'B'; else
        if( pos.mWhiteRooks   & bit )   *cur++ = 'R'; else
        if( pos.mWhiteQueens  & bit )   *cur++ = 'Q'; else
        if( pos.mWhiteKing    & bit )   *cur++ = 'K'; else
        if( pos.mBlackPawns   & bit )   *cur++ = 'p'; else
        if( pos.mBlackKnights & bit )   *cur++ = 'n'; else
        if( pos.mBlackBishops & bit )   *cur++ = 'b'; else
        if( pos.mBlackRooks   & bit )   *cur++ = 'r'; else
        if( pos.mBlackQueens  & bit )   *cur++ = 'q'; else
        if( pos.mBlackKing    & bit )   *cur++ = 'k'; else
        {
            if( (cur > str) && (cur[-1] >= '1') && (cur[-1] < '8') )
                cur[-1]++;
            else
                *cur++ = '1';
        }

        if( bit & FILE_H )
            *cur++ = (bit & RANK_1)? ' ' : '/';
    }

    str = cur;

    *str++ = pos.mWhiteToMove? 'w' : 'b';
    *str++ = ' ';

    if(   pos.mCastlingAndEP & SQUARE_H1 )     *str++ = 'K';
    if(   pos.mCastlingAndEP & SQUARE_A1 )     *str++ = 'Q';
    if(   pos.mCastlingAndEP & SQUARE_H8 )     *str++ = 'k';
    if(   pos.mCastlingAndEP & SQUARE_A8 )     *str++ = 'q';
    if( !(pos.mCastlingAndEP & CASTLE_ROOKS) ) *str++ = '-';

    *str++ = ' ';

    if( pos.mCastlingAndEP & EP_SQUARES )
    {
        int idx = (int) LowestBitIndex( (u64) (pos.mCastlingAndEP & EP_SQUARES) );
        *str++ = "hgfedcba"[idx & 7];
        *str++ = "12345678"[idx / 8]; 
        *str++ = ' ';
    }
    else
    {
        *str++ = '-';
        *str++ = ' ';
    }

    sprintf( str, "%d %d", (int) pos.mHalfmoveClock, (int) pos.mFullmoveNum );
}
#endif


template< typename SIMD >
class GamePlayer
{
    enum { LANES = SimdWidth< SIMD >::LANES };

    const PlayoutParams* mParams;
    RandomGen mRandom;

public:

    PDECL GamePlayer( const PlayoutParams* params, int salt = 0 )
    {
        mParams = params;
        mRandom.SetSeed( params->mRandomSeed + salt );
    }

    PDECL void PlayGames( const Position* pos, ScoreCard* dest, int simdCount )
    {
        assert( (uintptr_t) pos % sizeof( SIMD ) == 0 );

        int totalIters = simdCount * mParams->mNumGamesEach;

        #pragma omp parallel for schedule(dynamic) if (mParams->mEnableMulticore)
        for( int i = 0; i < totalIters; i++ )
        {
            PositionT< SIMD > simdPos;
            int idx = i % simdCount;

            Swizzle< SIMD >( pos + (idx * LANES), &simdPos );
            PlayOneGame( simdPos, dest + (idx * LANES) );
        }
    }

protected:

    PDECL void PlayOneGame( const PositionT< SIMD >& startPos, ScoreCard* outScores )
    {
        PositionT< SIMD > simdPos = startPos;
        MoveMapT< SIMD > simdMoveMap;
        simdPos.CalcMoveMap( &simdMoveMap );

        for( int i = 0; i < mParams->mMaxMovesPerGame; i++ )
        {
            MoveSpecT< SIMD > simdSpec = ChoosePlayoutMoves( simdPos, simdMoveMap );

            simdPos.Step( simdSpec, &simdMoveMap );
            if( GamesAreAllDone( simdPos ) )
                break;
        }

        u64* results = (u64*) &simdPos.mResult;
        for( int lane = 0; lane < LANES; lane++ )
        {
            outScores[lane].mWins[WHITE] += (results[lane] == RESULT_WHITE_WIN);
            outScores[lane].mWins[BLACK] += (results[lane] == RESULT_BLACK_WIN);
            outScores[lane].mPlays++;
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
        if( pos.mResult == RESULT_UNKNOWN )
        {
            MoveSpec randomMove = SelectRandomMove( pos, moveMap );
            return randomMove;
        }

        MoveSpec nullMove( 0, 0, 0 );
        return nullMove;
    }

    PDECL bool GamesAreAllDone( const PositionT< SIMD >& simdPos )
    {
        u64* results = (u64*) &simdPos.mResult;

        for( int i = 0; i < LANES; i++ )
            if( results[i] == RESULT_UNKNOWN )
                return false;

        return true;
    }

    PDECL MoveSpec SelectRandomMove( const Position& pos, const MoveMap& moveMap )
    {
        MoveList moveList;
        MoveMap sparseMap;
        u64* buf = (u64*) &sparseMap;

        // All the fields in the MoveMap (up to mCheckMask) represent moves as bits

        const int count = (int) offsetof( MoveMap, mCheckMask ) / sizeof( u64 );

        sparseMap = moveMap;

        // Choose a lucky random bit (move) to keep

        u64 total = 0;
        for( int i = 0; i < count; i++ )
            total += PlatCountBits64( buf[i] );

        assert( total > 0 );
        u64 bitsToSkip = mRandom.GetRange( total );

        // Find which word it's in

        int word = 0;
        while( word < count )
        {
            u64 bits = PlatCountBits64( buf[word] );
            if( bits >= bitsToSkip )
                break;

            bitsToSkip -= bits;
            buf[word++] = 0;
        }

        // Keep just that bit, and clear the rest

        u64 idx;
        u64 wordVal = buf[word];
        while( bitsToSkip-- )
            idx = ConsumeLowestBitIndex( wordVal );

        word++;
        while( word < count )
            buf[word++] = 0;

        moveList.UnpackMoveMap( pos, sparseMap );

        for( int i = 0; i < moveList.mCount; i++ )
            if( moveList.mMove[i].mDest == idx )
                return moveList.mMove[i];

        assert( moveList.mCount == 0 );

        static u64 sCounts[MAX_POSSIBLE_MOVES] = { 0 };
        sCounts[moveList.mCount]++;

        if( moveList.mCount == 0 )
        {
            moveList.UnpackMoveMap( pos, moveMap );
        }

        u64 idx = mRandom.GetRange( moveList.mCount );
        return moveList.mMove[idx];
    }


};

