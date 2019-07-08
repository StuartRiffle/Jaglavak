// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"

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

            assert( expansionBranch[i]->_Node == NULL );
            TreeNode* newNode = _SearchTree->CreateBranch( node, expansionBranchIdx );
            assert( expansionBranch[i]->_Node == newNode );

            _Batch->_Position.push_back( newNode->_Pos );
        }

        BatchRef ourBatch = _Batch;

        if( (_Batch->_Position.size() >= _Settings->Get( "Search.BatchSize" )) || _Settings->Get( "Search.FlushEveryBatch" ) )
            this->FlushBatch();

        while( !ourBatch->_Done )
        {
            _SearchFibers.YieldFiber();
            ourBatch->_YieldCounter++;
        }

        cout << "YIELD COMPLETE" << endl;

        assert( ourBatch->_GameResults.size() >= (offset + count) );

        for( int i = 0; i < count; i++ )
        {
            ScoreCard& results = ourBatch->_GameResults[offset + i];
            assert( results._Plays == ourBatch->_Params._NumGamesEach );

            expansionBranch[i]->_Scores.Add( results );
            totalScore.Add( results );
        }

        _SearchTree->Touch( node );
        return totalScore;
    }

    int chosenBranchIdx = SelectNextBranch( node );
       if ( chosenBranchIdx< 0 )
           chosenBranchIdx = SelectNextBranch(node);
    assert( chosenBranchIdx >= 0 );

    BranchInfo* chosenBranch = &node->_Branch[chosenBranchIdx];

    ScoreCard branchScores = ExpandAtLeaf( chosenBranch->_Node, depth + 1 );
    chosenBranch->_Scores.Add( branchScores );

    _SearchTree->Touch( node );

    return branchScores;
}

void TreeSearch::FlushBatch()
{
    if( !_Batch )
        return;

    _BatchQueue.Push( _Batch );
    _Batch = NULL;

    _Metrics._NumBatchesMade++;
}




