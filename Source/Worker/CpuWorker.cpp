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
        _WorkThreads.emplace_back( new thread( [this]() { ___CPU_WORK_THREAD___(); } ) );

    return (_WorkThreads.size() > 0);
}

void CpuWorker::PrintCpuInfo()
{
    string cpuName   = CpuInfo::GetCpuName();
    int    cores     = CpuInfo::DetectCpuCores();
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

void CpuWorker::___CPU_WORK_THREAD___()
{
    while( !_TimeToExit )
    {
        BatchRef batch;
        if( !_BatchQueue->Pop( batch ) )
            break;

        int count = ( int) batch->_Position.size();
        batch->_GameResults.resize( count + SIMD_WIDEST );

        PlayGamesCpu(
            _Settings,
            &batch->_Params,
            batch->_Position.data(),
            batch->_GameResults.data(),
            count );

        batch->_GameResults.resize( count );  
        batch->_Done = true;
    }
}

