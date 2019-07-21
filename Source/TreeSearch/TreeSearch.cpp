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
    u64 seed = _Settings->Get( "Debug.ForceRandomSeed" );
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
    shared_ptr< CpuWorker > cpuWorker( new CpuWorker( _Settings, &_Metrics, &_BatchQueue ) );
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

            shared_ptr< CudaWorker > cudaWorker( new CudaWorker( _Settings, &_Metrics, &_BatchQueue ) );
            if( cudaWorker->Initialize( i ) )
                _Workers.push_back( cudaWorker );
        }
    }
}

TreeNode* SearchTree::FollowMoveList( TreeNode* node, const MoveList& moveList, int idx )
{
    if( idx >= moveList._Count )
        return node;

    int branchIdx = node->FindMoveIndex( moveList._Move[idx] );
    if( branchIdx < 0 )
        return NULL;

    BranchInfo& info = node->_Branch[branchIdx];
    if( info._Node )
        return FollowMoveList( info._Node, moveList, idx + 1 );

    return NULL;
}

void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList )
{
    this->StopSearching();

    /*
    if( pos == _GameStartPosition )
    {
        if( moveList._Count >= _GameHistory.size() )
        {
            if( !memcmp( moveList._Move, _GameHistory._Move, _GameHistory->_Count * sizeof( MoveSpec ) ) )
            {
                TreeNode* rootNode = _SearchTree->GetRootNode();
                TreeNode* existingChild = FollowMoveList( rootNode, moveList, _GameHistory._Count );
                if( existingChild )
                {
                    _SearchRoot = existingChild;
                    _RootInfo = *(existingChild->_Info);
                    _SearchRoot->_Info = &_RootInfo;

                    cout << "info string Using cached position with " << _RootInfo._Scores._Plays " games";

                    _GameHistory = moveList;
                    return;
                }
            }
        }
    }
    */

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
    _ExplorationFactor = _Settings->Get< float >( "Search.ExplorationFactor" );

    _PlayoutParams._RandomSeed      = _RandomGen.GetNext();
    _PlayoutParams._NumGamesEach    = _Settings->Get( "Search.NumPlayoutsEach" );
    _PlayoutParams._MaxMovesPerGame = _Settings->Get( "Search.MaxPlayoutMoves" );
    _PlayoutParams._LimitPlayoutCores        = _Settings->Get( "CPU.LimitPlayoutCores" );
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

    _SearchThread = unique_ptr< thread >( new thread( [this] { this->___SEARCH_THREAD___(); } ) );
}
                                         
void TreeSearch::StopSearching()
{
    if( _SearchThread )
    {
        _SearchExit = true;
        _SearchThread->join();
        _SearchThread = NULL;
        _SearchExit = false;
    }
}

void TreeSearch::___SEARCH_THREAD___()
{
    PlatSetThreadName( "_SEARCH" );

    _SearchTimer.Reset();

    while( !_SearchExit )
    {
        if( IsTimeToMove() )
            break;

        for( auto& worker : _Workers )
            worker->Update();

        this->UpdateFibers();
        this->UpdateUciStatus();
    } 

    SendUciBestMove();
    _SearchFibers.TerminateAll();
}

void TreeSearch::___SEARCH_FIBER___()
{
    TreeNode* root = _SearchTree->GetRootNode();

    ScoreCard rootScores = this->ExpandAtLeaf( root );
    root->_Info->_Scores.Add( rootScores );
}

static int sMostFibers = 0;

void TreeSearch::UpdateFibers()
{
    _SearchFibers.UpdateAll();

    if( _SearchExit )
        return;

    int fiberLimit = _Settings->Get( "CPU.SearchFibers" );
    if( _SearchFibers.GetCount() < fiberLimit )
        _SearchFibers.Spawn( [&]() { this->___SEARCH_FIBER___(); } );

    sMostFibers = MAX( sMostFibers, _SearchFibers.GetCount() );
}

void TreeSearch::UpdateUciStatus()
{
    if( _UciUpdateTimer.GetElapsedMs() >= _Settings->Get( "UCI.UpdateTime" ) )
    {
        cout << endl;
        SendUciStatus();
        _UciUpdateTimer.Reset();

        cout << endl;
        _SearchTree->DumpRoot();

        cout << endl;
        _SearchTree->DumpTop();
    }
}




