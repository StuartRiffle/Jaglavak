// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess/Core.h"
#include "Common.h"
#include "FEN.h"
#include "TreeSearch.h"

#include "Player/GamePlayer.h"
#include "Worker/CpuWorker.h"
#include "Worker/CudaWorker.h"
#include "Util/FiberSet.h"

TreeSearch::TreeSearch( GlobalSettings* settings ) : 
    _Settings( settings )
{
    u64 seed = _Settings["FixedRandomSeed"];
    if( seed == 0 )
        seed = CpuInfo::GetClockTick();

    _RandomGen.SetSeed( seed );

    _SearchTree->Init();
    this->Reset();
}

void TreeSearch::Init()
{
    shared_ptr< CpuWorker > cpuWorker( new CpuWorker( _Settings, &_BatchQueue ) );
    if( cpuWorker->Initialize() )
        _Workers.push_back( cpuWorker );

    if( _Settings["EnableCuda"] )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            int mask = _Settings["GpuAffinityMask"];
            if( mask != 0 )
                if( ((1 << i) & mask) == 0 )
                    continue;

            shared_ptr< CudaWorker > cudaWorker( new CudaWorker( _Settings, &_BatchQueue ) );
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

        if( fibers.GetCount() < _Settings["MaxSearchFibers"] )
            fibers.Spawn( [&]() { this->SearchFiber(); } );            
    } 
}

void TreeSearch::SearchFiber()
{
    TreeNode* root = _SearchTree->GetRootNode();

    ScoreCard rootScores = this->ExpandAtLeaf( root );
    root->_Info->_Scores.Add( rootScores );
}






