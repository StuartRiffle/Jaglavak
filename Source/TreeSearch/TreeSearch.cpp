// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "FEN.h"
#include "GamePlayer.h"

#include "SIMD/SimdWorker.h"
#include "CUDA/CudaWorker.h"

TreeSearch::TreeSearch( GlobalOptions* options, u64 randomSeed ) : 
    _Options( options )
{
    _SearchRoot = NULL;
    _ShuttingDown = false;
    _SearchingNow = false;
    _NumPending = 0;

    _Random.SetSeed( randomSeed );

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

void TreeSearch::Init()
{
    cout << "CPU: " << CpuInfo::GetCpuName() << " (" << CpuInfo::DetectCpuCores() << " cores, " << CpuInfo::DetectSimdLevel() << "x SIMD)" << endl;

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

            auto prop = worker->GetDeviceProperties();
            int totalCores = CudaWorker::GetCoresPerSM( prop.major, prop.minor );
            cout << "GPU: " << prop.name << " (" << totalCores << " cores)" << endl;

            _AsyncWorkers.push_back( worker );
        }
    }

    _SearchThread  = unique_ptr< thread >( new thread( [this] { this->SearchThread(); } ) );
    _SearchThreadIsIdle.Wait();
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
    _ShuttingDown = true;
    this->StopSearching();

    _DoneQueue.Terminate();
    _WorkQueue.Terminate();
    _AsyncWorkers.clear();

    _SearchThreadWakeUp.Post();
    _SearchThread->join();
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

void TreeSearch::CreateNewNode( MoveList& pathFromRoot, TreeNode* node, int branchIdx )
{
    TreeNode* newNode = AllocNode();
    assert( newNode != node );

    BranchInfo* chosenBranch = &node->_Branch[branchIdx];

    MoveToFront( node );

    MoveMap newMap;
    Position newPos = node->_Pos;
    newPos.Step( chosenBranch->_Move, &newMap );

    newNode->InitPosition( newPos, newMap, chosenBranch ); 
    this->CalculatePriors( newNode, pathFromRoot );

    chosenBranch->_Node = newNode;
}



void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList )
{
    this->StopSearching();

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





ScoreCard TreeSearch::ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node, BatchRef batch )
{
    MoveToFront( node );

    if( node->_GameOver )
        return( node->_GameResult );

    int numUnexpanded = 0;
    for( int i = 0; i < (int) node->_Branch.size(); i++ )
        if( node->_Branch[i]._Node == NULL )
            numUnexpanded++;

    if( numUnexpanded > 0 )
    {
        ScoreCard totalScore;

        int numToExpand = numUnexpanded;
        if( _Options->_MaxBranchExpansion )
            if( numToExpand > _Options->_MaxBranchExpansion )
                numToExpand = _Options->_MaxBranchExpansion;

        Position ALIGN_SIMD pos[MAX_POSSIBLE_MOVES];

        for( int i = 0; i < numToExpand; i++ )
        {
            int expansionBranchIdx = SelectNextBranch( node );
            BranchInfo* expansionBranch = &node->_Branch[expansionBranchIdx];

            assert( expansionBranch->_Node == NULL );

            MoveList pathToBranch = pathFromRoot;
            pathToBranch.Append( expansionBranch->_Move );

            this->CreateNewNode( pathToBranch, node, expansionBranchIdx );

            _DeepestLevelSearched = Max( _DeepestLevelSearched, pathFromRoot._Count );

            if( _SearchParams._InitialPlayouts )
                pos[i] = expansionBranch->_Node->_Pos;

            if( _SearchParams._AsyncPlayouts )
            {
                batch->Append( expansionBranch->_Node->_Pos, pathToBranch );

                // IMPORTANT
                // FIXME: explain why
                totalScore._Plays += _SearchParams._AsyncPlayouts;
            }
        }

        if( _SearchParams._InitialPlayouts )
        {
            PlayoutParams playoutParams = batch->_Params;
            playoutParams._NumGamesEach = _SearchParams._InitialPlayouts;

            ScoreCard scores[MAX_POSSIBLE_MOVES];
            PlayGamesSimd( _Options, &playoutParams, pos, scores, numToExpand );

            for( int i = 0; i < numToExpand; i++ )
                totalScore.Add( scores[i] );
        }

        return totalScore;
    }

    int chosenBranchIdx = SelectNextBranch( node );
    BranchInfo* chosenBranch = &node->_Branch[chosenBranchIdx];
 
    ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, chosenBranch->_Node, batch );
    chosenBranch->_Scores.Add( branchScores );

    MoveToFront( node );

    return branchScores;
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


