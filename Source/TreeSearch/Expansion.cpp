// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"
#include "CpuPlayer.h"

ScoreCard TreeSearch::ExpandAtLeaf( TreeNode* node, int depth )
{
    TreeNode::RefScope protect( node );

    _SearchTree->Touch( node );

    if( node->_GameOver )
        return( node->_GameResult );

    _DeepestLevelSearched = MAX( depth, _DeepestLevelSearched );

    int unexpanded = 0;
    for( auto& info : node->_Branch )
        if( info._Node == NULL )
            unexpanded++;

    if( unexpanded > 0 )
    {
        ScoreCard totalScore;

        int count = _Settings->Get( "Search.BranchesToExpand" );
        if( count == 0 )
            count = unexpanded;

        if( !_Batch )
        {
            _Batch = BatchRef( new PlayoutBatch() );
            _Batch->_Params = _PlayoutParams;
        }

        BranchInfo* expansionBranch[MAX_POSSIBLE_MOVES];
        size_t offset = _Batch->_Position.size();

        for( int i = 0; i < count; i++ )
        {
            int expansionBranchIdx = SelectNextBranch( node );
            expansionBranch[i] = &node->_Branch[expansionBranchIdx];

            TreeNode* newNode = _SearchTree->CreateBranch( node, expansionBranchIdx );
            EstimatePriors( newNode );
            _Metrics._NodesExpanded++;

            _Batch->_Position.push_back( newNode->_Pos );
        }

        BatchRef ourBatch = _Batch;

        int doPlayoutsNow = _Settings->Get( "Debug.DoPlayoutsInline" );
        if( doPlayoutsNow )
        {
            PlayBatchCpu( _Settings, ourBatch );
            ourBatch->_Done = true;
        }
        else
        {
            int batchFullEnough = (_Batch->_Position.size() >= _Settings->Get( "Search.BatchSize" ));
            int forceFlush = _Settings->Get( "Debug.FlushEveryBatch" );

            if( batchFullEnough || forceFlush )
                FlushBatch();

            while( !ourBatch->_Done )
            {
                // ------------------------
                _SearchFibers.YieldFiber();
                // ------------------------

                ourBatch->_YieldCounter++;
                if( _SearchExit )
                    return totalScore;
            }
        }

        assert( ourBatch->_Done );
        assert( ourBatch->_GameResults.size() >= (offset + count) );

        for( int i = 0; i < count; i++ )
        {
            ScoreCard& results = ourBatch->_GameResults[offset + i];
            //assert( results._Plays == ourBatch->_Params._NumGamesEach );

            expansionBranch[i]->_Scores.Add( results );
            totalScore.Add( results );
        }
    
        _SearchTree->Touch( node );
        return totalScore;
    }

    int chosenBranchIdx = SelectNextBranch( node );
    assert( chosenBranchIdx >= 0 );

    BranchInfo* chosenBranch = &node->_Branch[chosenBranchIdx];

    BranchInfo::VirtualLossScope lossScope( *chosenBranch, _Settings->Get< float >( "Search.VirtualLoss" ) );

    ScoreCard branchScores = ExpandAtLeaf( chosenBranch->_Node, depth + 1 );
    chosenBranch->_Scores.Add( branchScores );

    _SearchTree->Touch( node );
    return branchScores;
}

void TreeSearch::FlushBatch()
{
    if( _Batch )
    {
        _Batch->_TickQueued = CpuInfo::GetClockTick();
        _BatchQueue.Push( _Batch );
        _Batch = NULL;

        _Metrics._BatchesQueued++;
    }
}






