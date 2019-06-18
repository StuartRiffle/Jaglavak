// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "SimdPlayer.h"

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

class CpuWorker : public AsyncWorker
{
    const GlobalOptions*    _Options;
    BatchQueue*             _WorkQueue;
    BatchQueue*             _DoneQueue;
    unique_ptr< thread >    _WorkThread;

    void WorkThread()
    {
        for( ;; )
        {
            BatchRef batch;
            if( !_WorkQueue->Pop( batch ) )
                break;

            int count = (int) batch->_Position.size();
            batch->_GameResults.resize( count + SIMD_MAX );

            PlayGamesSimd( 
                _Options, 
                &batch->_Params, 
                batch->_Position.data(), 
                batch->_GameResults.data(),
                count );

            batch->_GameResults.resize( count );
            batch->_Done = true;
        }
    }

public:

    SimdWorker( const GlobalOptions* options, BatchQueue* queue )
    {
        _Options = options;
        _Queue   = queue;
        _WorkThread = unique_ptr< thread >( new thread( [this] { this->WorkThread(); } ) );
    }

    ~SimdWorker()
    {
        _WorkThread->join();
    }
};
