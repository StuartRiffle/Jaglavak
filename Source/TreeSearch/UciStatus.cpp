// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"
#include "Util/FEN.h"

void TreeSearch::ExtractBestLine( TreeNode* node, MoveList* dest )
{
    if( !node )
        return;

    u64 bestPlays = 0;
    int bestPlaysIdx = -1;
    int numBranches = (int) node->_Branch.size();

    for( int i = 0; i < numBranches; i++ )
    {
        if( !node->_Branch[i]._Node )
            continue;

        u64 branchPlays = (u64) node->_Branch[i]._Scores._Plays;
        if( branchPlays > bestPlays )
        {
            bestPlays = branchPlays;
            bestPlaysIdx = i;
        }
    }

    if( bestPlaysIdx < 0 )
        return;

    const BranchInfo& branchInfo = node->_Branch[bestPlaysIdx];

    dest->Append( branchInfo._Move );
    ExtractBestLine( branchInfo._Node, dest );
}

int TreeSearch::EstimatePawnAdvantageForMove( const MoveSpec& spec )
{
    TreeNode* root = _SearchTree->GetRootNode();

    int bestIdx = root->FindMoveIndex( spec );
    assert( bestIdx >= 0 );

    int color = root->_Info->_Node->_Color;
    BranchInfo& info = root->_Branch[bestIdx];

    float winRatio = 0;
    if( info._Scores._Plays > 0 )
        winRatio = info._Scores._Wins[color] * 1.0f / info._Scores._Plays;

    float pawnAdvantage = log10f( winRatio / (1 - winRatio) ) * 4;
    int centipawns = (int) (pawnAdvantage * 100);

    if( !root->_Info->_Node->_Pos._WhiteToMove )
        centipawns *= -1;

    return centipawns;
}

MoveSpec TreeSearch::SendUciStatus()
{
    if( _Metrics._BatchesDone < 1 )
        return MoveSpec();

    float dt = _UciUpdateTimer.GetElapsedSec();

    u64 nodesDone = _Metrics._NodesExpanded - _StatsStartMetrics._NodesExpanded;
    u64 nodesPerSec = (u64) (nodesDone / dt);

    u64 batchesDone = _Metrics._BatchesDone - _StatsStartMetrics._BatchesDone;
    u64 batchesPerSec = (u64) (batchesDone / dt);

    u64 gamesDone = _Metrics._GamesPlayed - _StatsStartMetrics._GamesPlayed;
    u64 gamesPerSec = (u64) (gamesDone / dt);

    double batchLatency = _Metrics._BatchTotalLatency * 1.0 / (_Metrics._BatchesDone * CpuInfo::GetClockFrequency());
    double batchRuntime = _Metrics._BatchTotalRuntime * 1.0 / (_Metrics._BatchesDone * CpuInfo::GetClockFrequency());

    MoveList bestLine;
    ExtractBestLine( _SearchTree->GetRootNode(), &bestLine );

    int evaluation = 0;
    if( bestLine._Count > 0 )
        evaluation = EstimatePawnAdvantageForMove( bestLine._Move[0] );

    cout << "info"  <<
        " nodes "   << _Metrics._NodesExpanded <<
        " nps "     << nodesPerSec <<
        " cp "      << evaluation <<
        " depth "   << _DeepestLevelSearched <<
        " time "    << _SearchTimer.GetElapsedMs() <<
        " pv "      << SerializeMoveList( bestLine ) <<
        endl;

    cout << "info string" <<
        " games "   << _Metrics._GamesPlayed <<
        " gps "     << gamesPerSec <<
        " batches " << _Metrics._BatchesDone <<
        " bps "     << batchesPerSec <<
        " latency " << batchLatency <<
        " runtime " << batchRuntime <<
        endl;

    _StatsStartMetrics = _Metrics;

    return bestLine._Move[0];
}


void TreeSearch::SendUciBestMove()
{
    MoveList bestLine;
    ExtractBestLine( _SearchTree->GetRootNode(), &bestLine );

    if( bestLine._Count > 0 )
    {
        printf( "bestmove %s", SerializeMoveSpec( bestLine._Move[0] ).c_str() );
        if( bestLine._Count > 1 )
            printf( " ponder %s", SerializeMoveSpec( bestLine._Move[1] ).c_str() );
    }
    _SearchTree->DumpTop();

    printf( "\n" );
}
