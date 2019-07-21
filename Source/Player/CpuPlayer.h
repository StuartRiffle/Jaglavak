// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

static void PlayGamesCpu( const GlobalSettings* settings, const PlayoutParams* params, const Position* pos, ScoreCard* dest, int count )
{
    int simdLevel = CpuInfo::GetSimdLevel();

    if( !settings->Get( "CPU.SIMD.Enabled" ) )
        simdLevel = 1;

    int forceLevel = settings->Get( "CPU.SIMD.ForceLevel" );
    if( forceLevel )
        simdLevel = forceLevel;

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

static void PlayBatchCpu( const GlobalSettings* settings, BatchRef& batch )
{
    int count = (int) batch->_Position.size();
    batch->_GameResults.resize( count + SIMD_WIDEST );

    PlayGamesCpu(
        settings,
        &batch->_Params,
        batch->_Position.data(),
        batch->_GameResults.data(),
        count );

    batch->_GameResults.resize( count );
    batch->_Done = true;
}


