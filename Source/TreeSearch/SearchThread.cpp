// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"

void TreeSearch::SetUciSearchConfig( const UciSearchConfig& config )
{
    _UciConfig = config;
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    _SearchStartMetrics = _Metrics;

    _SearchingNow = true;
    _SearchThreadWakeUp.Post();
}

void TreeSearch::StopSearching()
{
    if( _SearchingNow )
    {
        _SearchingNow = false;
        _SearchThreadIsIdle.Wait();
    }
}

void TreeSearch::DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& scores, int depth )
{
    if( depth >= pathFromRoot._Count )
        return;

    MoveSpec move = pathFromRoot._Move[depth];

    int childIdx = node->FindMoveIndex( move );
    assert( childIdx >= 0 );

    BranchInfo& childInfo = node->_Branch[childIdx];

    TreeNode* child = childInfo._Node;
    assert( child );

    DeliverScores( child, pathFromRoot, scores, depth + 1 );

    childInfo._Scores._Wins[WHITE] += scores._Wins[WHITE];
    childInfo._Scores._Wins[BLACK] += scores._Wins[BLACK];
    // _Plays already credited when scheduling batch
    // FIXME doc this

    //childInfo._Scores += scores;

    #if DEBUG   
    childInfo._DebugLossCounter--;
    #endif
}

void TreeSearch::ProcessScoreBatch( BatchRef& batch )
{
    #if DEBUG_VALIDATE_BATCH_RESULTS
    // Extremely expensive: every batch is recalculated to verify
    for( int i = 0; i < batch->GetCount(); i++ )
    {
        ScoreCard checkScores;
        int salt = i;

        GamePlayer< u64 > player( &batch->_Params, salt );
        player.PlayGames( &batch->_Position[i], &checkScores, 1 );

        assert( checkScores._Wins[0] == batch->_GameResults[i]._Wins[0] );
        assert( checkScores._Wins[1] == batch->_GameResults[i]._Wins[1] );
        assert( checkScores._Plays   == batch->_GameResults[i]._Plays );
    }
    #endif

    for( int i = 0; i < batch->GetCount(); i++ )
    {
        this->DeliverScores( _SearchRoot, batch->_PathFromRoot[i], batch->_GameResults[i] );
        _Metrics._NumGamesPlayed += batch->_GameResults[i]._Plays;
    }

    _Metrics._NumBatchesDone++;
}

void TreeSearch::SearchThread()
{
    for( ;; )
    {
        _SearchThreadIsIdle.Post();
        _SearchThreadWakeUp.Wait();

        if( _ShuttingDown )
            return;

        _SearchParams._BatchSize       = _Options->_BatchSize;
        _SearchParams._MaxPending      = _Options->_MaxPendingBatches;
        _SearchParams._AsyncPlayouts   = _Options->_NumAsyncPlayouts;
        _SearchParams._InitialPlayouts = _Options->_NumInitialPlayouts;

        _SearchTimer.Reset();
        while( _SearchingNow )
        {
            if( IsTimeToMove() )
                break;

            for( auto& worker : _AsyncWorkers )
                worker->Update();

            for( auto& batch : _DoneQueue.PopAll() )
            {
                ProcessScoreBatch( batch );
                _NumPending--;
            }

            if( _Metrics._NumBatchesDone > 0 )
                if( _UciUpdateTimer.GetElapsedMs() >= _Options->_UciUpdateDelay )
                    SendUciStatus();

            if( _NumPending >= _SearchParams._MaxPending )
            {
                PlatSleep( _Options->_SearchSleepTime );
                continue;
            }

            auto batch = CreateNewBatch();
            if( batch->GetCount() > 0 )
            {
                _WorkQueue.Push( batch );
                _NumPending++;
                _Metrics._NumBatchesMade++;
            }
        }

        MoveList bestLine;
        ExtractBestLine( _SearchRoot, &bestLine );

        cout << "bestmove " << SerializeMoveSpec( bestLine._Move[0] );
        if( bestLine._Count > 1 )
            cout << " ponder " << SerializeMoveSpec( bestLine._Move[1] );
        cout << endl;
    }
}

