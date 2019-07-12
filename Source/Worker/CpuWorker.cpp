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
        _WorkThreads.emplace_back( new thread( [this]() { WorkThread(); } ) );

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

    const char* ampersand = strchr( cpuName.c_str(), '@' );
    if( ampersand )
        cpuName = cpuName.substr( 0, ampersand - cpuName.c_str() - 1 );

    cout << "[CPU] " << cpuName << endl;

    if( ampersand )
    {
        string clockSpeed = (ampersand + 1);
        trim( clockSpeed );

        cout << "[CPU]   " << clockSpeed << endl;
    }

    cout << "[CPU]   " << cores << " total cores" << endl;
    cout << "[CPU] SIMD   " << simdDesc << endl;
    cout << "[CPU] Lanes  " << simdLevel << endl;


}

void CpuWorker::WorkThread()
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

