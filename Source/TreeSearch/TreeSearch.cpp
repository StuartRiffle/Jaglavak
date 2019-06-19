// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"
#include "GamePlayer.h"

#include "Worker/CpuWorker.h"
#include "Worker/CudaWorker.h"

TreeSearch::TreeSearch( GlobalOptions* options ) : 
    _Options( options )
{
    u64 seed = CpuInfo::GetClockTick();
    if( _Options->_FixedRandomSeed )
        seed = _Options->_FixedRandomSeed;
    _RandomGen.SetSeed( seed );

    _SearchTree->Init();
    this->Reset();
}

void TreeSearch::Init()
{
    shared_ptr< CpuWorker > cpuWorker( new CpuWorker( _Options, &_BatchQueue ) );
    if( cpuWorker->Initialize() )
        _Workers.push_back( cpuWorker );

    if( _Options->_EnableCuda )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            if( _Options->_GpuAffinityMask )
                if( ((1 << i) & _Options->_GpuAffinityMask) == 0 )
                    continue;

            shared_ptr< CudaWorker > cudaWorker( new CudaWorker( _Options, &_BatchQueue ) );
            if( cudaWorker->Initialize( i ) )
                _Workers.push_back( cudaWorker );
        }
    }
}

void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList )
{
    this->StopSearching();

    _SearchTree->SetPosition( startPos, moveList );
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
    this->StopSearching();

    _BatchQueue.Terminate();
    _Workers.clear();
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    _SearchTimer.Reset();
    _SearchThread = unique_ptr< thread >( new thread( [this] { this->SearchThread(); } ) );
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

void TreeSearch::SearchThread()
{
    FiberSet fibers;

    while( !_SearchExit )
    {
        if( IsTimeToMove() )
            break;

        for( auto& worker : _Workers )
            worker->Update();

        fibers.Update();

        if( fibers.GetCount() < _Options->_MaxSearchFibers )
            fibers.Spawn( [&]() { this->SearchFiber(); } );            
    } 
}

void TreeSearch::SearchFiber()
{
    TreeNode* root = _SearchTree->GetRootNode();

    ScoreCard rootScores = this->ExpandAtLeaf( root );
    root->_Info->_Scores.Add( rootScores );
}






