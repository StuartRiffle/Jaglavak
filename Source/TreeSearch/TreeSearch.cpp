// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"
#include "GamePlayer.h"

#include "SIMD/SimdWorker.h"
#include "CUDA/CudaWorker.h"

TreeSearch::TreeSearch( GlobalOptions* options, u64 rando_Seed ) : 
    _Options( options )
{
    _SearchRoot = NULL;
    _ShuttingDown = false;
    _SearchingNow = false;
    _NumPending = 0;

    _Random.SetSeed( rando_Seed );

    _NodePoolEntries = _Options->_MaxTreeNodes;
    _NodePoolBuf.resize( _NodePoolEntries * sizeof( TreeNode ) + SIMD_ALIGNMENT );
    _NodePool = (TreeNode*) (((uintptr_t) _NodePoolBuf.data() + SIMD_ALIGNMENT - 1) & ~(SIMD_ALIGNMENT - 1));

    for( int i = 0; i < _NodePoolEntries; i++ )
    {
        _NodePool[i]._Prev = &_NodePool[i - 1];
        _NodePool[i]._Next = &_NodePool[i + 1];
    }

    _NodePool[0]._Prev = (TreeNode*) &_MruListHead;
    _MruListHead._Next = &_NodePool[0];

    _NodePool[_NodePoolEntries - 1]._Next = (TreeNode*) &_MruListHead;
    _MruListHead._Prev = &_NodePool[_NodePoolEntries - 1];

    this->Reset();
}

TreeSearch::~TreeSearch()
{
    _ShuttingDown = true;
    this->StopSearching();

    _DoneQueue.Terminate();
    _WorkQueue.Terminate();
    _AsyncWorkers.clear();

    _SearchThreadGo.Post();
    _SearchThread->join();
}

void TreeSearch::Init()
{
    for( int i = 0; i < _Options->_NumSimdWorkers; i++ )
    {
        auto worker = new SimdWorker( _Options, &_WorkQueue, &_DoneQueue );
        _AsyncWorkers.push_back( shared_ptr< AsyncWorker >( worker ) );
    }

    if( _Options->_EnableCuda )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            if( _Options->_GpuAffinityMask )
                if( ((1 << i) & _Options->_GpuAffinityMask) == 0 )
                    continue;

            shared_ptr< CudaWorker > worker( new CudaWorker( _Options, &_WorkQueue, &_DoneQueue ) );
            worker->Initialize( i );

            _AsyncWorkers.push_back( worker );
        }
    }

    _SearchThread  = unique_ptr< thread >( new thread( [this] { this->SearchThread(); } ) );
}

void TreeSearch::Reset()
{
    this->StopSearching();

    Position startPos;
    startPos.Reset();

    this->SetPosition( startPos );

    _Metrics.Clear();
    _SearchStartMetrics.Clear();
    _StatsStartMetrics.Clear();

    _DeepestLevelSearched = 0;
}

void TreeSearch::SetUciSearchConfig( const UciSearchConfig& config )
{
    _UciConfig = config;
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    _SearchStartMetrics = _Metrics;

    _SearchingNow = true;
    _SearchThreadGo.Post();
}

void TreeSearch::StopSearching()
{
    if( _SearchingNow )
    {
        _SearchingNow = false;
        _SearchThreadIsIdle.Wait();
    }
}



void TreeSearch::MoveToFront( TreeNode* node )
{
    TreeNode* oldFront = _MruListHead._Next;

    assert( node->_Next->_Prev == node );
    assert( node->_Prev->_Next == node );
    assert( oldFront->_Prev == (TreeNode*) &_MruListHead );

    node->_Next->_Prev = node->_Prev;
    node->_Prev->_Next = node->_Next;

    node->_Next = _MruListHead._Next;
    node->_Next->_Prev = node;

    node->_Prev = (TreeNode*) &_MruListHead;
    node->_Prev->_Next = node;

    assert( _MruListHead._Next == node );
}

TreeNode* TreeSearch::AllocNode()
{
    TreeNode* node = _MruListHead._Prev;

    node->Clear();
    MoveToFront( node );

    _Metrics._NumNodesCreated++;
    return node;
}

void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList )
{
    // TODO: recognize position and don't terf the whole tree

    Position pos = startPos;

    if( moveList )
        for( int i = 0; i < moveList->_Count; i++ )
            pos.Step( moveList->_Move[i] );

    MoveMap moveMap;
    pos.CalcMoveMap( &moveMap );

    if( _SearchRoot )
        _SearchRoot->_Info = NULL;

    _SearchRoot = AllocNode();
    _SearchRoot->InitPosition( pos, moveMap );

    _SearchRoot->_Info = &_RootInfo;
    _RootInfo._Node = _SearchRoot;
}

void TreeSearch::CalculatePriors( TreeNode* node, MoveList& pathFromRoot )
{
    // TODO
}

double TreeSearch::CalculateUct( TreeNode* node, int childIndex )
{
    BranchInfo* nodeInfo    = node->_Info;
    BranchInfo& childInfo   = node->_Branch[childIndex];
    const ScoreCard& scores = childInfo._Scores;

    u64 nodePlays  = Max< u64 >( nodeInfo->_Scores._Plays, 1 ); 
    u64 childPlays = Max< u64 >( scores._Plays, 1 );
    u64 childWins  = scores._Wins[node->_Color];
    
    if( _Options->_DrawsWorthHalf )
    {
        u64 draws = scores._Plays - (scores._Wins[WHITE] + scores._Wins[BLACK]);
        childWins += draws / 2;
    }

    double invChildPlays = 1.0 / childPlays;
    double childWinRatio = childWins * invChildPlays;

    double uct = 
        childWinRatio + 
        sqrt( log( (double) nodePlays ) * 2 * invChildPlays ) * _Options->_ExplorationFactor +
        childInfo._Prior;

    return uct;
}

int TreeSearch::GetRando_UnexploredBranch( TreeNode* node )
{
    int nu_Branches = (int) node->_Branch.size();
    int idx = (int) _Random.GetRange( nu_Branches );

    for( int i = 0; i < nu_Branches; i++ )
    {
        if( !node->_Branch[idx]._Node )
            return idx;

        idx = (idx + 1) % nu_Branches;
    }

    return( -1 );
}

int TreeSearch::SelectNextBranch( TreeNode* node )
{
    int nu_Branches = (int) node->_Branch.size();
    assert( nu_Branches > 0 );

    int rando_Branch = GetRando_UnexploredBranch( node );
    if( rando_Branch >= 0 )
        return rando_Branch;

    // This node is fully expanded, so choose the move with highest UCT

    double highestUct = DBL_MIN;
    int highestIdx = 0;

    for( int i = 0; i < nu_Branches; i++ )
    {
        double uct = CalculateUct( node, i );
        if( uct > highestUct )
        {
            highestUct = uct;
            highestIdx = i;
        }
    }

    return highestIdx;
}

ScoreCard TreeSearch::ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node, BatchRef batch )
{
    MoveToFront( node );

    if( node->_GameOver )
        return( node->_GameResult );

    int chosenBranchIdx = SelectNextBranch( node );
    BranchInfo* chosenBranch = &node->_Branch[chosenBranchIdx];

    assert( chosenBranch->_Prior == 0 );

    pathFromRoot.Append( chosenBranch->_Move );
    _DeepestLevelSearched = Max( _DeepestLevelSearched, pathFromRoot._Count );

#if DEBUG   
    chosenBranch->_DebugLossCounter++;
#endif

    if( !chosenBranch->_Node )
    {
        // This is a leaf, so create a new node 

        TreeNode* newNode = AllocNode();
        assert( newNode != node );

        MoveToFront( node );

        MoveMap newMap;
        Position newPos = node->_Pos;
        newPos.Step( chosenBranch->_Move, &newMap );

        newNode->InitPosition( newPos, newMap, chosenBranch ); 
        this->CalculatePriors( newNode, pathFromRoot );

        chosenBranch->_Node = newNode;

        if( newNode->_GameOver )
        {
            newNode->_Info->_Scores.Add( newNode->_GameResult );
            return( newNode->_GameResult );
        }

        ScoreCard scores;

        if( _SearchParams._InitialPlayouts > 0 )
        {
            PlayoutParams playoutParams = batch->_Params;
            playoutParams._NumGamesEach = _SearchParams._InitialPlayouts;

            GamePlayer< u64 > player( &playoutParams, (int) _Random.GetNext() );
            player.PlayGames( &newPos, &scores, 1 );            
        }

        if( _SearchParams._AsyncPlayouts > 0 )
        {
            batch->Append( newPos, pathFromRoot );
            scores._Plays += batch->_Params._NumGamesEach;
        }

        newNode->_Info->_Scores.Add( scores );
        return scores;
    }

    ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, chosenBranch->_Node, batch );

    // Addulate the scores on our way back down the tree

    chosenBranch->_Scores.Add( branchScores );

    // Mark each node MRU on the way

    MoveToFront( node );

    return branchScores;
}


bool TreeSearch::IsTimeToMove()
{
    const float MS_TO_SEC = 0.001f;

    bool    whiteToMove     = _SearchRoot->_Pos._WhiteToMove; 
    int     requiredMoves   = _UciConfig._TimeControlMoves;
    float   timeBuffer      = _Options->_TimeSafetyBuffer * MS_TO_SEC;
    float   timeElapsed     = _SearchTimer.GetElapsedSec() + timeBuffer;
    float   timeInc         = (whiteToMove? _UciConfig._WhiteTimeInc  : _UciConfig._BlackTimeInc)  * MS_TO_SEC;
    float   timeLeftAtStart = (whiteToMove? _UciConfig._WhiteTimeLeft : _UciConfig._BlackTimeLeft) * MS_TO_SEC;
    float   timeLimit       = _UciConfig._TimeLimit * MS_TO_SEC;
    float   timeLeft        = timeLeftAtStart - timeElapsed;

    if( timeLimit > 0 )
        if( timeElapsed > timeLimit )
            return true;

    if( (requiredMoves > 0) && (timeLeftAtStart > 0) )
        if( timeElapsed >= (timeLeftAtStart / requiredMoves) )
            return true;

    if( _UciConfig._NodesLimit > 0 )
        if( _Metrics._NumNodesCreated >= _UciConfig._NodesLimit )
            return true;

    if( _UciConfig._DepthLimit > 0 )
        if( _DeepestLevelSearched > _UciConfig._DepthLimit )
            return true;

    return false;
}


void TreeSearch::DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& scores, int depth )
{
    if( depth >= pathFromRoot._Count )
        return;

    MoveSpec move = pathFromRoot._Move[depth];

    int childIdx = node->FindMoveIndex( move );
    if( childIdx < 0 )
        return; // FIXME: should never happen

    BranchInfo& childInfo = node->_Branch[childIdx];

    TreeNode* child = childInfo._Node;
    if( child == NULL )
        return; // FIXME: should never happen

    DeliverScores( child, pathFromRoot, scores, depth + 1 );


    childInfo._Scores._Wins[WHITE] += scores._Wins[WHITE];
    childInfo._Scores._Wins[BLACK] += scores._Wins[BLACK];
    // _Plays already credited when scheduling batch

    //childInfo._Scores += scores;

#if DEBUG   
    childInfo._DebugLossCounter--;
#endif
}

void TreeSearch::ProcessScoreBatch( BatchRef& batch )
{
#if DEBUG_VALIDATE_BATCH_RESULTS
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

BatchRef TreeSearch::CreateNewBatch()
{
    BatchRef batch( new PlayoutBatch() );

    batch->_Params._RandomSeed      = _Random.GetNext();
    batch->_Params._NumGamesEach    = _Options->_NumAsyncPlayouts;
    batch->_Params._MaxMovesPerGame = _Options->_MaxPlayoutMoves;
    batch->_Params._EnableMulticore = _Options->_EnableMulticore;

    for( ;; )
    {
        MoveList pathFromRoot;
        ScoreCard rootScores = this->ExpandAtLeaf( pathFromRoot, _SearchRoot, batch );
        _SearchRoot->_Info->_Scores.Add( rootScores );

        if( _SearchParams._AsyncPlayouts == 0 )
            break;

        if( batch->GetCount() >= _SearchParams._BatchSize )
            break;
    }

    return batch;
}


void TreeSearch::SearchThread()
{
    for( ;; )
    {
        _SearchThreadIsIdle.Post();
        _SearchThreadGo.Wait();

        if( _ShuttingDown )
            return;

        _SearchParams._BatchSize       = _Options->_BatchSize;
        _SearchParams._MaxPending      = _Options->_MaxPendingBatches;
        _SearchParams._AsyncPlayouts   = _Options->_NumAsyncPlayouts;
        _SearchParams._InitialPlayouts = _Options->_NumInitialPlayouts;






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

void TreeSearch::ExtractBestLine( TreeNode* node, MoveList* dest )
{
    if( !node )
        return;

    u64 bestPlays = 0;
    int bestPlaysIdx = -1;
    int nu_Branches = (int) node->_Branch.size();

    for( int i = 0; i < nu_Branches; i++ )
    {
        if( !node->_Branch[i]._Node )
            return;

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

MoveSpec TreeSearch::SendUciStatus()
{
    float dt = _UciUpdateTimer.GetElapsedSec();

    u64 nodesDone = _Metrics._NumNodesCreated - _StatsStartMetrics._NumNodesCreated;
    u64 nodesPerSec = (u64) (nodesDone / dt);

    u64 batchesDone = _Metrics._NumBatchesDone - _StatsStartMetrics._NumBatchesDone;
    u64 batchesPerSec = (u64) (batchesDone / dt);

    u64 gamesDone = _Metrics._NumGamesPlayed - _StatsStartMetrics._NumGamesPlayed;
    u64 gamesPerSec = (u64) (gamesDone / dt);

    MoveList bestLine;
    ExtractBestLine( _SearchRoot, &bestLine );

    cout << "info" <<
        " nps " <<   nodesPerSec <<
        " bps " <<   batchesPerSec <<
        " gps " <<   gamesPerSec <<
        " depth " << _DeepestLevelSearched <<
        " nodes " << _Metrics._NumNodesCreated <<
        " time " <<  _SearchTimer.GetElapsedMs() <<
        " pv " <<    SerializeMoveList( bestLine ) <<
        endl;

    _StatsStartMetrics = _Metrics;

    _UciUpdateTimer.Reset();
    return bestLine._Move[0];
}


