// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

static void PlayGamesSimd( const GlobalOptions* options, const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    int simdLevel = options->_DetectedSimdLevel;

    if( !options->_EnableSimd )
        simdLevel = 1;

    if( options->_ForceSimdLevel )
        simdLevel = options->_ForceSimdLevel;

    int simdCount = (count + simdLevel - 1) / simdLevel;

    extern void PlayGamesAVX512( const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
    extern void PlayGamesAVX2(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
    extern void PlayGamesSSE4(   const PlayoutParams* params, const Position* pos, ScoreCard* dest, int simdCount );
    extern void PlayGamesX64(    const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count );

    switch( simdLevel )
    {
    case 8:   PlayGamesAVX512( params, pos, dest, simdCount ); break;
    case 4:   PlayGamesAVX2(   params, pos, dest, simdCount ); break;
    case 2:   PlayGamesSSE4(   params, pos, dest, simdCount ); break;
    default:  PlayGamesX64(    params, pos, dest, count ); break;
    }
}
