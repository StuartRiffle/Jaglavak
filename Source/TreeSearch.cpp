// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"
#include "CpuWorker.h"
#include "CudaSupport.h"
#include "CudaWorker.h"
#include "FEN.h"
#include "GamePlayer.h"

TreeSearch::TreeSearch( GlobalOptions* options, u64 randomSeed ) : 
    mOptions( options )
{
    mSearchRoot = NULL;
    mShuttingDown = false;
    mSearchingNow = false;

    mRandom.SetSeed( randomSeed );

    mNodePoolEntries = mOptions->mMaxTreeNodes;
    mNodePool = new TreeNode[mNodePoolEntries];

    for( int i = 0; i < mNodePoolEntries; i++ )
    {
        mNodePool[i].mPrev = &mNodePool[i - 1];
        mNodePool[i].mNext = &mNodePool[i + 1];
    }

    mNodePool[0].mPrev = (TreeNode*) &mMruListHead;
    mMruListHead.mNext = &mNodePool[0];

    mNodePool[mNodePoolEntries - 1].mNext = (TreeNode*) &mMruListHead;
    mMruListHead.mPrev = &mNodePool[mNodePoolEntries - 1];

    this->Reset();
}

TreeSearch::~TreeSearch()
{
    mShuttingDown = true;
    this->StopSearching();

    mDoneQueue.Terminate();
    mWorkQueue.Terminate();
    mAsyncWorkers.clear();

    mSearchThreadGo.Post();
    mSearchThread->join();

    delete[] mNodePool;
}

void TreeSearch::Init()
{
    for( int i = 0; i < mOptions->mNumCpuWorkers; i++ )
    {
        auto worker = new CpuWorker( mOptions, &mWorkQueue, &mDoneQueue );
        mAsyncWorkers.push_back( shared_ptr< AsyncWorker >( worker ) );
    }

    if( mOptions->mEnableCuda )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            // FIXME make a mask parameter
            if( i == 1 )
                continue;

            shared_ptr< CudaWorker > worker( new CudaWorker( mOptions, &mWorkQueue, &mDoneQueue ) );
            worker->Initialize( i, mOptions->mCudaQueueDepth );

            int warpSize = worker->GetDeviceProperties().warpSize;
            if( mOptions->mMinBatchSize < warpSize )
                mOptions->mMinBatchSize = warpSize;

            mAsyncWorkers.push_back( worker );
        }
    }

    mSearchThread  = unique_ptr< thread >( new thread( [this] { this->SearchThread(); } ) );
}

void TreeSearch::Reset()
{
    this->StopSearching();

    Position startPos;
    startPos.Reset();

    this->SetPosition( startPos );

    mMetrics.Clear();
    mSearchStartMetrics.Clear();
    mStatsStartMetrics.Clear();

    mDeepestLevelSearched = 0;
}

void TreeSearch::SetUciSearchConfig( const UciSearchConfig& config )
{
    mUciConfig = config;
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    mSearchStartMetrics = mMetrics;

    mSearchingNow = true;
    mSearchThreadGo.Post();
}

void TreeSearch::StopSearching()
{
    if( mSearchingNow )
    {
        mSearchingNow = false;
        mSearchThreadIsIdle.Wait();
    }
}



void TreeSearch::MoveToFront( TreeNode* node )
{
    TreeNode* oldFront = mMruListHead.mNext;

    assert( node->mNext->mPrev == node );
    assert( node->mPrev->mNext == node );
    assert( oldFront->mPrev == (TreeNode*) &mMruListHead );

    node->mNext->mPrev = node->mPrev;
    node->mPrev->mNext = node->mNext;

    node->mNext = mMruListHead.mNext;
    node->mNext->mPrev = node;

    node->mPrev = (TreeNode*) &mMruListHead;
    node->mPrev->mNext = node;

    assert( mMruListHead.mNext == node );
}

TreeNode* TreeSearch::AllocNode()
{
    TreeNode* node = mMruListHead.mPrev;

    node->Clear();
    MoveToFront( node );

    mMetrics.mNumNodesCreated++;
    return node;
}

void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList )
{
    // TODO: recognize position and don't terf the whole tree

    Position pos = startPos;

    if( moveList )
        for( int i = 0; i < moveList->mCount; i++ )
            pos.Step( moveList->mMove[i] );

    MoveMap moveMap;
    pos.CalcMoveMap( &moveMap );

    if( mSearchRoot )
        mSearchRoot->mInfo = NULL;

    mSearchRoot = AllocNode();
    mSearchRoot->InitPosition( pos, moveMap );

    mSearchRoot->mInfo = &mRootInfo;
    mRootInfo.mNode = mSearchRoot;
}

void TreeSearch::CalculatePriors( TreeNode* node, MoveList& pathFromRoot )
{
    // TODO
}

double TreeSearch::CalculateUct( TreeNode* node, int childIndex )
{
    BranchInfo* nodeInfo    = node->mInfo;
    BranchInfo& childInfo   = node->mBranch[childIndex];
    const ScoreCard& scores = childInfo.mScores;

    u64 nodePlays  = Max< u64 >( nodeInfo->mScores.mPlays, 1 ); 
    u64 childPlays = Max< u64 >( scores.mPlays, 1 );
    u64 childWins  = scores.mWins[node->mColor];
    
    if( mOptions->mDrawsWorthHalf )
    {
        u64 draws = scores.mPlays - (scores.mWins[WHITE] + scores.mWins[BLACK]);
        childWins += draws / 2;
    }

    double invChildPlays = 1.0 / childPlays;
    double childWinRatio = childWins * invChildPlays;

    double uct = 
        childWinRatio + 
        sqrt( log( nodePlays ) * 2 * invChildPlays ) * mOptions->mExplorationFactor +
        childInfo.mPrior;

    assert( childInfo.mPrior == 0 );

    return uct;
}

int TreeSearch::GetRandomUnexploredBranch( TreeNode* node )
{
    int numBranches = (int) node->mBranch.size();
    int idx = (int) mRandom.GetRange( numBranches );

    for( int i = 0; i < numBranches; i++ )
    {
        if( !node->mBranch[idx].mNode )
            return idx;

        idx = (idx + 1) % numBranches;
    }

    return( -1 );
}

int TreeSearch::SelectNextBranch( TreeNode* node )
{
    int numBranches = (int) node->mBranch.size();
    assert( numBranches > 0 );

    int randomBranch = GetRandomUnexploredBranch( node );
    if( randomBranch >= 0 )
        return randomBranch;

    // This node is fully expanded, so choose the move with highest UCT

    double highestUct = DBL_MIN;
    int highestIdx = 0;

    for( int i = 0; i < numBranches; i++ )
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

    if( node->mGameOver )
        return( node->mGameResult );

    int chosenBranchIdx = SelectNextBranch( node );
    BranchInfo* chosenBranch = &node->mBranch[chosenBranchIdx];

    assert( chosenBranch->mPrior == 0 );

    pathFromRoot.Append( chosenBranch->mMove );
    mDeepestLevelSearched = Max( mDeepestLevelSearched, pathFromRoot.mCount );

#if DEBUG   
    chosenBranch->mDebugLossCounter++;
#endif

    if( !chosenBranch->mNode )
    {
        // This is a leaf, so create a new node 

        TreeNode* newNode = AllocNode();
        assert( newNode != node );

        MoveToFront( node );

        MoveMap newMap;
        Position newPos = node->mPos;
        newPos.Step( chosenBranch->mMove, &newMap );

        newNode->InitPosition( newPos, newMap, chosenBranch ); 
        this->CalculatePriors( newNode, pathFromRoot );

        chosenBranch->mNode = newNode;

        if( newNode->mGameOver )
        {
            newNode->mInfo->mScores += newNode->mGameResult;
            return( newNode->mGameResult );
        }

        ScoreCard scores;

        if( mSearchParams.mInitialPlayouts > 0 )
        {
            PlayoutParams playoutParams = batch->mParams;
            playoutParams.mNumGamesEach = mSearchParams.mInitialPlayouts;

            GamePlayer< u64 > player( &playoutParams, mRandom.GetNext() );
            player.PlayGames( &newPos, &scores, 1 );            
        }

        if( mSearchParams.mAsyncPlayouts > 0 )
        {
            batch->Append( newPos, pathFromRoot );
            scores.mPlays += batch->mParams.mNumGamesEach;
        }

        newNode->mInfo->mScores += scores;
        return scores;
    }

    ScoreCard branchScores = ExpandAtLeaf( pathFromRoot, chosenBranch->mNode, batch );

    // Accumulate the scores on our way back down the tree

    chosenBranch->mScores += branchScores;

    // Mark each node MRU on the way

    MoveToFront( node );

    return branchScores;
}


bool TreeSearch::IsTimeToMove()
{
    const float MS_TO_SEC = 0.001f;

    bool    whiteToMove     = mSearchRoot->mPos.mWhiteToMove; 
    int     requiredMoves   = mUciConfig.mTimeControlMoves;
    float   timeBuffer      = mOptions->mTimeSafetyBuffer * MS_TO_SEC;
    float   timeElapsed     = mSearchTimer.GetElapsedSec() + timeBuffer;
    float   timeInc         = (whiteToMove? mUciConfig.mWhiteTimeInc  : mUciConfig.mBlackTimeInc)  * MS_TO_SEC;
    float   timeLeftAtStart = (whiteToMove? mUciConfig.mWhiteTimeLeft : mUciConfig.mBlackTimeLeft) * MS_TO_SEC;
    float   timeLimit       = mUciConfig.mTimeLimit * MS_TO_SEC;
    float   timeLeft        = timeLeftAtStart - timeElapsed;

    if( timeLimit > 0 )
        if( timeElapsed > timeLimit )
            return true;

    if( (requiredMoves > 0) && (timeLeftAtStart > 0) )
        if( timeElapsed >= (timeLeftAtStart / requiredMoves) )
            return true;

    if( mUciConfig.mNodesLimit > 0 )
        if( mMetrics.mNumNodesCreated >= mUciConfig.mNodesLimit )
            return true;

    if( mUciConfig.mDepthLimit > 0 )
        if( mDeepestLevelSearched > mUciConfig.mDepthLimit )
            return true;

    return false;
}


void TreeSearch::DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& scores, int depth )
{
    if( depth >= pathFromRoot.mCount )
        return;

    MoveSpec move = pathFromRoot.mMove[depth];

    int childIdx = node->FindMoveIndex( move );
    if( childIdx < 0 )
        return; // FIXME: should never happen

    BranchInfo& childInfo = node->mBranch[childIdx];

    TreeNode* child = childInfo.mNode;
    if( child == NULL )
        return; // FIXME: should never happen

    DeliverScores( child, pathFromRoot, scores, depth + 1 );


    childInfo.mScores.mWins[WHITE] += scores.mWins[WHITE];
    childInfo.mScores.mWins[BLACK] += scores.mWins[BLACK];
    // mPlays already credited when scheduling batch

    //childInfo.mScores += scores;

#if DEBUG   
    childInfo.mDebugLossCounter--;
#endif
}

void TreeSearch::ProcessScoreBatch( BatchRef& batch )
{
#if DEBUG_VALIDATE_BATCH_RESULTS
    for( int i = 0; i < batch->GetCount(); i++ )
    {
        ScoreCard checkScores;
        int salt = i;

        GamePlayer< u64 > player( &batch->mParams, salt );
        player.PlayGames( &batch->mPosition[i], &checkScores, 1 );

        assert( checkScores.mWins[0] == batch->mResults[i].mWins[0] );
        assert( checkScores.mWins[1] == batch->mResults[i].mWins[1] );
        assert( checkScores.mPlays   == batch->mResults[i].mPlays );
    }
#endif

    for( int i = 0; i < batch->GetCount(); i++ )
    {
        this->DeliverScores( mSearchRoot, batch->mPathFromRoot[i], batch->mResults[i] );
        mMetrics.mNumGamesPlayed += batch->mResults[i].mPlays;
    }

    mMetrics.mNumBatchesDone++;
}

PlayoutParams TreeSearch::GetPlayoutParams()
{
    PlayoutParams params = { 0 };

    params.mRandomSeed      = mRandom.GetNext();
    params.mNumGamesEach    = mOptions->mMaxAsyncPlayouts;
    params.mMaxMovesPerGame = mOptions->mMaxPlayoutMoves;
    params.mEnableMulticore = mOptions->mEnableMulticore;

    return params;
}

BatchRef TreeSearch::CreateNewBatch()
{
    BatchRef batch( new PlayoutBatch );
    batch->mParams = this->GetPlayoutParams();

    for( ;; )
    {
        MoveList pathFromRoot;
        ScoreCard rootScores = this->ExpandAtLeaf( pathFromRoot, mSearchRoot, batch );
        mSearchRoot->mInfo->mScores += rootScores;

        if( mSearchParams.mAsyncPlayouts == 0 )
        {
            assert( batch->GetCount() == 0 );
            break;
        }

        int batchLimit = Min( mSearchParams.mBatchSize, PLAYOUT_BATCH_MAX );
        if( batch->GetCount() >= batchLimit )
            break;
    }

    return batch;
}


void TreeSearch::AdjustForWarmup()
{
	bool warmingUp = (mMetrics.mNumBatchesDone < mOptions->mNumWarmupBatches);

    mSearchParams.mBatchSize       = warmingUp? mOptions->mMinBatchSize      : mOptions->mMaxBatchSize;
    mSearchParams.mMaxPending      = warmingUp? mOptions->mMinPendingBatches : mOptions->mMaxPendingBatches;
    mSearchParams.mAsyncPlayouts   = warmingUp? mOptions->mMinAsyncPlayouts  : mOptions->mMaxAsyncPlayouts;
	mSearchParams.mInitialPlayouts = mOptions->mNumInitialPlayouts;

    if( mSearchParams.mInitialPlayouts + mSearchParams.mAsyncPlayouts == 0 )
        mSearchParams.mAsyncPlayouts = 1;
}

void TreeSearch::SearchThread()
{
    for( ;; )
    {
        mSearchThreadIsIdle.Post();
        mSearchThreadGo.Wait();

        if( mShuttingDown )
            return;
            
        mSearchTimer.Reset();
        while( mSearchingNow )
        {
            AdjustForWarmup();

            if( IsTimeToMove() )
                break;

            for( auto& worker : mAsyncWorkers )
                worker->Update();

            for( auto& batch : mDoneQueue.PopMulti() )
                ProcessScoreBatch( batch );

            if( mMetrics.mNumBatchesDone > 0 )
                if( mUciUpdateTimer.GetElapsedMs() >= mOptions->mUciUpdateDelay )
                    SendUciStatus();

            if( mWorkQueue.PeekCount() >= mSearchParams.mMaxPending )
            {
                PlatSleep( mOptions->mSearchSleepTime );
                continue;
            }

            auto batch = CreateNewBatch();
            if( batch->GetCount() > 0 )
            {
                mWorkQueue.Push( batch );
                mMetrics.mNumBatchesMade++;
            }
        }

        MoveSpec bestMove = SendUciStatus();
        printf( "bestmove %s\n", SerializeMoveSpec( bestMove ).c_str() );
    }
}

void TreeSearch::ExtractBestLine( TreeNode* node, MoveList* dest )
{
    u64 bestPlays = 0;
    int bestPlaysIdx = -1;
    int numBranches = (int) node->mBranch.size();

    for( int i = 0; i < numBranches; i++ )
    {
        if( !node->mBranch[i].mNode )
            return;

        u64 branchPlays = (u64) node->mBranch[i].mScores.mPlays;
        if( branchPlays > bestPlays )
        {
            bestPlays = branchPlays;
            bestPlaysIdx = i;
        }
    }

    assert( bestPlaysIdx >= 0 );
    const BranchInfo& branchInfo = node->mBranch[bestPlaysIdx];

    dest->Append( branchInfo.mMove );
    ExtractBestLine( branchInfo.mNode, dest );
}

MoveSpec TreeSearch::SendUciStatus()
{
    float dt = mUciUpdateTimer.GetElapsedSec();

    u64 nodesDone = mMetrics.mNumNodesCreated - mStatsStartMetrics.mNumNodesCreated;
    u64 nodesPerSec = (u64) (nodesDone / dt);

    u64 batchesDone = mMetrics.mNumBatchesDone - mStatsStartMetrics.mNumBatchesDone;
    u64 batchesPerSec = (u64) (batchesDone / dt);

    u64 gamesDone = mMetrics.mNumGamesPlayed - mStatsStartMetrics.mNumGamesPlayed;
    u64 gamesPerSec = (u64) (gamesDone / dt);

    MoveList bestLine;
    ExtractBestLine( mSearchRoot, &bestLine );

    printf( "info" );
    printf( " nps %" PRId64 "   ", nodesPerSec );
    printf( " bps %" PRId64, batchesPerSec );
    printf( " gps %" PRId64, gamesPerSec );

    printf( " depth %d", mDeepestLevelSearched );
//    printf( " nodes %" PRId64, mMetrics.mNumNodesCreated );
//    printf( " batches %" PRId64, mMetrics.mNumBatchesDone );
    printf( " time %d", mSearchTimer.GetElapsedMs() ),
    printf( " pv %s", SerializeMoveList( bestLine ).c_str() );
    printf( "\n" );

    mStatsStartMetrics = mMetrics;

    mUciUpdateTimer.Reset();
    return bestLine.mMove[0];
}


