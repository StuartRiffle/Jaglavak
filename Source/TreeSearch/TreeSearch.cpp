// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"
#include "GamePlayer.h"

#include "SIMD/SimdWorker.h"
#include "CUDA/CudaWorker.h"

TreeSearch::TreeSearch( GlobalOptions* options ) : 
    _Options( options )
{
    _SearchRoot = NULL;
    _ShuttingDown = false;
    _SearchingNow = false;
    _NumPending = 0;

    u64 seed = CpuInfo::GetClockTick();
    if( _Options->FixedRandomSeed )
        seed = _Options->FixedRandomSeed;

    _Random.SetSeed( seed );

    _Tree->CreateNodePool();

    this->Reset();
}

void TreeSearch::Init()
{
    shared_ptr< AsyncWorker > cpuWorker( new CpuWorker( _Options, &_BatchQueue ) );
    if( cpuWorker->Initialize() )
        _Workers.push_back( cpuWorker ) );

    if( _Options->_EnableCuda )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            if( _Options->_GpuAffinityMask )
                if( ((1 << i) & _Options->_GpuAffinityMask) == 0 )
                    continue;

            shared_ptr< AsyncWorker > cudaWorker( new CudaWorker( _Options, &_BatchQueue ) );
            if( cudaWorker->Initialize( i ) )
                _Workers.push_back( cudaWorker );
        }
    }
}

void TreeSearch::Reset()
{
    this->StopSearching();

    _Metrics.Clear();
    _SearchStartMetrics.Clear();
    _StatsStartMetrics.Clear();

    _DeepestLevelSearched = 0;
    _GameHistory.Clear();

    Position startPos;
    startPos.Reset();
    this->SetPosition( startPos );
}

TreeSearch::~TreeSearch()
{
    _ShuttingDown = true;
    this->StopSearching();

    _BatchQueue.Terminate();
    _AsyncWorkers.clear();
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    _SearchTimer.Reset();
    _SearchThread  = unique_ptr< thread >( new thread( [this] { this->SearchThread(); } ) );
}
                                         
void TreeSearch::StopSearching()
{
    if( _SearchThread )
    {
        _SearchExit = true;
        _SearchThread->join();

        _SearchExit = false;
        _SearchThread = NULL;
    }
}

void Search::SearchThread()
{
    FiberSet fibers;

    while( !_SearchExit )
    {
        if( IsTimeToMove() )
            break;

        for( auto& worker : _Workers )
            worker->Update();

        fibers.Update();

        if( fibers.GetNumRunning() < _Options->MaxSearchFibers )
            fibers.Spawn( [&]() { this->SearchFiber(); } );            
    } 
}

void Search::SearchFiber()
{
    ScoreCard rootScores = this->ExpandAtLeaf( _SearchRoot );
    _SearchRoot->_Info->_Scores.Add( rootScores );
}







