// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "CpuWorker.h"
#include "CpuPlayer.h"

#include "boost/algorithm/string.hpp"
using namespace boost;

bool CpuWorker::Initialize()
{
    PrintCpuInfo();

    for( int i = 0; i < _Settings->Get( "CPU.DispatchThreads" ); i++ )
        _WorkThreads.emplace_back( new thread( [=,this]() { ___CPU_WORK_THREAD___( i ); } ) );

    return (_WorkThreads.size() > 0);
}

void CpuWorker::PrintCpuInfo()
{
    int    cores     = PlatDetectCpuCores();
    string cpuName   = CpuInfo::GetCpuName();
    int    simdLevel = CpuInfo::GetSimdLevel();
    string simdDesc  = CpuInfo::GetSimdDesc( simdLevel );

    replace_all( cpuName, "(R)", "" );
    replace_all( cpuName, "CPU ", "" );

    string clockSpeed;

    size_t ampersand = cpuName.find( "@" );
    if( ampersand != string::npos )
    {
        clockSpeed = cpuName.substr( ampersand + 1 );
        cpuName = cpuName.substr( 0, ampersand - 1 );
    }

    cout << "CPU: " << cpuName << endl;

    if( !clockSpeed.empty() )
        cout << "  Clock   " << clockSpeed << endl;

    cout << "  SIMD     " << simdDesc << " (" << simdLevel << "x)" << endl;
    cout << "  Cores    " << cores << endl;
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

