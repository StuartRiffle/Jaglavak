// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"

#include "Player/GamePlayer.h"
#include "Worker/CpuWorker.h"
#include "Worker/CudaWorker.h"
#include "Util/FEN.h"
#include "Util/FiberSet.h"

TreeSearch::TreeSearch( GlobalSettings* settings ) : 
    _Settings( settings )
{
    u64 seed = _Settings->Get( "Search.ForceRandomSeed" );
    if( seed == 0 )
        seed = CpuInfo::GetClockTick();

    _RandomGen.SetSeed( seed );

    _SearchTree = unique_ptr< SearchTree >( new SearchTree( settings ) );
    _SearchTree->Init();

    this->Reset();

    Position startPos;
    startPos.Reset();
    this->SetPosition( startPos );
}

void TreeSearch::Init()
{
    shared_ptr< CpuWorker > cpuWorker( new CpuWorker( _Settings, &_BatchQueue ) );
    if( cpuWorker->Initialize() )
        _Workers.push_back( cpuWorker );

    if( _Settings->Get( "CUDA.Enabled" ) )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            int mask = _Settings->Get( "CUDA.AffinityMask" );
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
    assert( _SearchThread == NULL );

    // TODO: recognize position and don't terf the whole tree

    Position pos = startPos;
    if( moveList )
        for( int i = 0; i < moveList->_Count; i++ )
            pos.Step( moveList->_Move[i] );

    _SearchTree->SetPosition( pos );
}

void TreeSearch::Reset()
{
    this->StopSearching();

    _Metrics.Clear();
    _SearchStartMetrics.Clear();
    _StatsStartMetrics.Clear();
    _GameHistory.Clear();

    _DeepestLevelSearched = 0;

    _DrawsWorthHalf    = _Settings->Get( "Search.DrawsWorthHalf" );
    _ExplorationFactor = _Settings->Get( "Search.ExplorationFactor" ) / 100.0f;

    _PlayoutParams._RandomSeed      = _RandomGen.GetNext();
    _PlayoutParams._NumGamesEach    = _Settings->Get( "Search.NumPlayouts" );
    _PlayoutParams._MaxMovesPerGame = _Settings->Get( "Search.NumPlayoutMoves" );
    _PlayoutParams._Multicore       = _Settings->Get( "CPU.Multicore" );

}

TreeSearch::~TreeSearch()
{
    this->StopSearching();

    _BatchQueue.Terminate();
    _Workers.clear();
}

void TreeSearch::StartSearching()
{
    this->Reset();
    assert( _SearchThread == NULL );

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

    _SearchTimer.Reset();
    while( !_SearchExit )
    {
        if( IsTimeToMove() )
            break;

        for( auto& worker : _Workers )
            worker->Update();

        if( _UciUpdateTimer.GetElapsedMs() >= _Settings->Get( "UCI.UpdateTime" ) )
        {
            _UciUpdateTimer.Reset();
            SendUciStatus();
        }

        int fiberLimit = _Settings->Get( "CPU.SearchFibers" );
        if( fiberLimit > 1 )
        {
            fibers.Update();

            if( fibers.GetCount() < fiberLimit )
                fibers.Spawn( [&]() { this->SearchFiber(); } );            
        }
        else
        {
            // Call synchronously (for debugging)

            this->SearchFiber();
        }
    } 
}

void TreeSearch::SearchFiber()
{
    TreeNode* root = _SearchTree->GetRootNode();

    ScoreCard rootScores = this->ExpandAtLeaf( root );
    root->_Info->_Scores.Add( rootScores );
}






