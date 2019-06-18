// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

ScoreCard TreeSearch::ExpandAtLeaf( TreeNode* node )
{
    TreeNode::RefScope( node );

    MoveToFront( node );

    if( node->_GameOver )
        return( node->_GameResult );

    int unexpanded = 0;
    for( auto& info : node->_Branch )
        if( info._Node == NULL )
            unexpanded++;

    if( unexpanded > 0 )
    {
        ScoreCard totalScore;

        int count = numUnexpanded;
        if( _Options->_MaxBranchExpansion )
            count = CLAMP( count, _Options->_MaxBranchExpansion );

        if( !_Batch )
        {
            _Batch = new PlayoutBatch();

            _Batch->_Params._RandomSeed      = _Random.GetNext();
            _Batch->_Params._NumGamesEach    = _Options->_NumAsyncPlayouts;
            _Batch->_Params._MaxMovesPerGame = _Options->_MaxPlayoutMoves;
            _Batch->_Params._EnableMulticore = _Options->_EnableMulticore;
        }

        BranchInfo* expansionBranch[MAX_POSSIBLE_MOVES];
        size_t offset = _Batch->_Position.size();

        for( int i = 0; i < count; i++ )
        {
            int expansionBranchIdx = SelectNextBranch( node );
            expansionBranch[i] = &node->_Branch[expansionBranchIdx];
            assert( expansionBranch[i]->_Node == NULL );

            this->CreateNewNode( node, expansionBranchIdx );

            assert( expansionBranch[i]->_Node != NULL );
            _Batch->_Position.push_back( expansionBranch[i]->_Node->Pos );
        }

        BatchRef ourBatch = _Batch;

        if( (_Batch->_Position.size() >= _Options->_BatchSize) || _Options->FlushEveryBatch )
            this->FlushBatch();

        while( !ourBatch->_Done )
            YIELD_FIBER();

        assert( batch->_GameResults.size() >= (offset + count) );

        for( int i = 0; i < count; i++ )
        {
            ScoreCard& results = batch->_GameResults[offset + i];
            assert( results._Plays == _Batch->_Params._NumGamesEach );

            expansionBranch[i]->_Scores.Add( results );
            totalScore.Add( results );
        }

        return totalScore;
    }

    int chosenBranchIdx = SelectNextBranch( node );
    assert( chosenBranchIdx >= 0 );

    BranchInfo* chosenBranch = &node->_Branch[chosenBranchIdx];

    ScoreCard branchScores = ExpandAtLeaf( chosenBranch->_Node );
    chosenBranch->_Scores.Add( branchScores );

    MoveToFront( node );

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




