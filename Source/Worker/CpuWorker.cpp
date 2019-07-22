// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "CpuWorker.h"
#include "CpuPlayer.h"

#include "boost/algorithm/string.hpp"
using namespace boost;

bool CpuWorker::Initialize()
{
    for( int i = 0; i < _Settings->Get( "CPU.DispatchThreads" ); i++ )
        _WorkThreads.emplace_back( new thread( [=,this]() { ___CPU_WORK_THREAD___( i ); } ) );

    return (_WorkThreads.size() > 0);
}

string CpuWorker::GetDesc()
{
    int    cores     = PlatDetectCpuCores();
    string cpuName   = CpuInfo::GetCpuName();
    int    simdLevel = CpuInfo::GetSimdLevel();
    string simdDesc  = CpuInfo::GetSimdDesc( simdLevel );

    stringstream desc;
    desc << "CPU: " << cpuName << ", " << cores << " cores, " << simdDesc << " (" << simdLevel << "x)";

    return desc.str();
}


void CpuWorker::___CPU_WORK_THREAD___( int idx )
{
    string threadName = "_CPU " + idx;
    PlatSetThreadName( threadName.c_str() );

    while( !_TimeToExit )
    {
        BatchRef batch;
        if( !_BatchQueue->Pop( batch ) )
            break;

        Timer runTimer;
        PlayBatchCpu( _Settings, batch );

        u64 gamesPlayed = 0;
        for( int i = 0; i < batch->GetCount(); i++ )
        {
            assert( batch->_GameResults[i]._Plays > 0 );
            gamesPlayed += batch->_GameResults[i]._Plays;
        }

        _Metrics->_GamesPlayed += gamesPlayed;
        _Metrics->_BatchesDone++;
        _Metrics->_BatchTotalLatency += CpuInfo::GetClockTick() - batch->_TickQueued;
        _Metrics->_BatchTotalRuntime += runTimer.GetElapsedTicks();
    }
}

