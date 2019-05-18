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
    mBatchesDone = 0;
    mBatchesDoneThisSearch = 0;

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
    this->StopSearching();

    mShuttingDown = true;
    mSearchThreadGo.Post();
    mSearchThread->join();

    delete mNodePool;
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

    mBatchesMade = 0;
    mBatchesDone = 0;
    mSearchDepth = 0;
}

void TreeSearch::SetUciSearchConfig( const UciSearchConfig& config )
{
    mUciConfig = config;
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

    mBatchesMadeThisSearch = 0;
    mBatchesDoneThisSearch = 0;

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
    {
        node->mInfo->mScores += node->mGameResult;
        return( node->mGameResult );
    }

    int chosenBranchIdx = SelectNextBranch( node );
    BranchInfo* chosenBranch = &node->mBranch[chosenBranchIdx];

    assert( chosenBranch->mPrior == 0 );

    pathFromRoot.Append( chosenBranch->mMove );
    mDeepestLevel = Max( mDeepestLevel, pathFromRoot.mCount );

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

        if( mOptions->mNumInitialPlayouts > 0 )
        {
            PlayoutParams playoutParams = batch->mParams;
            playoutParams.mNumGamesEach = mOptions->mNumInitialPlayouts;

            GamePlayer< u64 > player( &playoutParams, mRandom.GetNext() );
            player.PlayGames( &newPos, &scores, 1 );            
        }

        if( mOptions->mNumAsyncPlayouts > 0 )
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



void TreeSearch::DumpStats( TreeNode* node )
{
    u64 bestPlays = 0;
    int bestPlaysIdx = 0;

    float bestRatio = 0;
    int bestRatioIdx = 0;

    for( int i = 0; i < (int) node->mBranch.size(); i++ )
    {
        if( node->mBranch[i].mScores.mPlays > bestPlays )
        {
            bestPlays = node->mBranch[i].mScores.mPlays;
            bestPlaysIdx = i;
        }

        if( node->mBranch[i].mScores.mPlays > 0 )
        {
            float ratio = node->mBranch[i].mScores.mWins[node->mColor] * 1.0f / node->mBranch[i].mScores.mPlays;

            if( ratio > bestRatio )
            {
                bestRatio = ratio;
                bestRatioIdx = i;
            }
        }
    }

    printf( "\n" );
    for( int i = 0; i < (int) node->mBranch.size(); i++ )
    {
        string moveText = SerializeMoveSpec( node->mBranch[i].mMove );
        printf( "%s%s  %2d) %5s %.15f %12ld/%-12ld\n", 
            (i == bestRatioIdx)? ">" : " ", 
            (i == bestPlaysIdx)? "***" : "   ", 
            i,
            moveText.c_str(), 
            this->CalculateUct( node, i ), 
            (u64) node->mBranch[i].mScores.mWins[node->mColor], (u64) node->mBranch[i].mScores.mPlays );
    }
}


bool TreeSearch::IsTimeToMove()
{
    const float MS_TO_SEC = 0.0001f;

    bool    whiteToMove     = mSearchRoot->mPos.mWhiteToMove; 
    int     requiredMoves   = mUciConfig.mTimeControlMoves;
    float   timeBuffer      = mOptions->mTimeBuffer * MS_TO_SEC;
    float   timeElapsed     = mSearchTimer.GetElapsedSec() + timeBuffer;
    float   timeInc         = (whiteToMove? mUciConfig.mWhiteTimeInc  : mUciConfig.mBlackTimeInc)  * MS_TO_SEC;
    float   timeLeftAtStart = (whiteToMove? mUciConfig.mWhiteTimeLeft : mUciConfig.mBlackTimeLeft) * MS_TO_SEC;
    float   timeLimit       = mUciConfig.mTimeLimit * MS_TO_SEC;
    float   timeLeft        = timeLeftAtStart - timeElapsed;
    u64     nodesDone       = mBatchesDoneThisSearch * mOptions->mBatchSize;

    if( timeLimit > 0 )
        if( timeLeft <= 0 )
            return true;

    if( requiredMoves > 0 )
        if( timeElapsed >= (timeLeft / requiredMoves) )
            return true;

    if( mUciConfig.mNodesLimit > 0 )
        if( nodesDone >= mUciConfig.mNodesLimit )
            return true;

    if( mUciConfig.mDepthLimit > 0 )
        if( mDeepestLevel >= mUciConfig.mDepthLimit )
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
        this->DeliverScores( mSearchRoot, batch->mPathFromRoot[i], batch->mResults[i] );

    mBatchesDoneThisSearch++;
    mBatchesDone++;
}

void TreeSearch::ProcessIncomingScores()
{
    for( ;; )
    {
        auto batches = mDoneQueue.PopMulti();
        if( batches.empty() )
            break;

        for( auto& batch : batches )
            this->ProcessScoreBatch( batch );
    }
}

void TreeSearch::UpdateAsyncWorkers()
{
}

PlayoutParams TreeSearch::GetPlayoutParams()
{
    PlayoutParams params = { 0 };

    params.mRandomSeed      = mRandom.GetNext();
    params.mNumGamesEach    = mOptions->mNumAsyncPlayouts;
    params.mMaxMovesPerGame = mOptions->mPlayoutMaxMoves;
    params.mEnableMulticore = mOptions->mEnableMulticore;

    return params;
}

BatchRef TreeSearch::ExpandTree()
{
    BatchRef batch( new PlayoutBatch );
    batch->mParams = this->GetPlayoutParams();

    int batchLimit = Min( mOptions->mBatchSize, PLAYOUT_BATCH_MAX );

    for( ;; )
    {
        MoveList pathFromRoot;
        ScoreCard rootScores = this->ExpandAtLeaf( pathFromRoot, mSearchRoot, batch );
        mSearchRoot->mInfo->mScores += rootScores;

        if( mOptions->mNumAsyncPlayouts == 0 )
        {
            assert( batch->GetCount() == 0 );
            break;
        }

        if( batch->GetCount() == batchLimit )
            break;
    }

    return batch;
}

void TreeSearch::SearchThread()
{
    for( ;; )
    {
        mSearchThreadIsIdle.Post();
        mSearchThreadGo.Wait();
        mSearchTimer.Reset();

        if( mShuttingDown )
            return;
            
        while( mSearchingNow )
        {
            if( IsTimeToMove() )
                break;

            for( auto& worker : mAsyncWorkers )
                worker->Update();
                
            ProcessIncomingScores();

            if( mUpdateTimer.GetElapsedSec() >= mOptions->mUciUpdateDelay )
                SendUciStatus();

            if( mWorkQueue.PeekCount() >= mOptions->mMaxPendingBatches )
            {
                PlatSleep( mOptions->mSearchSleepTime );
                continue;
            }

            auto batch = ExpandTree();
            if( batch->GetCount() > 0 )
            {
                mWorkQueue.Push( batch );
                mBatchesMadeThisSearch++;
                mBatchesMade++;
            }
        }

        MoveSpec bestMove = SendUciStatus();
        printf( "bestmove %s\n", SerializeMoveSpec( bestMove ) );
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

MoveSpec TreeSearch::SendUciStatus( bool printBestMove )
{
    u64 nodesDone = mBatchesDoneThisSearch * mOptions->mBatchSize;
    u64 nodesPerSec = (u64) (nodesDone / mSearchTimer.GetElapsedSec());

    MoveList bestLine;
    ExtractBestLine( mSearchRoot, &bestLine );

    printf( "info" );
    printf( " nps %12"PRId64 , nodesPerSec );
    printf( " depth %2d", mSearchDepth );
    printf( " nodes %12"PRId64, nodesDone );
    printf( " time %9d", mSearchTimer.GetElapsedMs() ),
    printf( " pv %s", SerializeMoveList( bestLine ).c_str() );
    printf( "\n" );

    if( printBestMove )
        printf( "bestmove %s\n", SerializeMoveSpec( bestLine.mMove[0] ) );

    mUpdateTimer.Reset();
}


