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
            // FIXME
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
}

void TreeSearch::SetUciSearchConfig( const UciSearchConfig& config )
{
    mUciConfig = config;
}

void TreeSearch::StartSearching()
{
    this->StopSearching();

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
        childInfo.mPrior -
        childInfo.mVirtualLoss;

    assert( childInfo.mPrior == 0 );

    return uct;
}

void TreeSearch::DecayVirtualLoss( TreeNode* node )
{
    for( int i = 0; i < (int) node->mBranch.size(); i++ )
        node->mBranch[i].mVirtualLoss *= mOptions->mVirtualLossDecay;
}

int TreeSearch::SelectNextBranch( TreeNode* node )
{
    this->DecayVirtualLoss( node );

    int numBranches = (int) node->mBranch.size();
    assert( numBranches > 0 );

    // Choose an untried branch at random

    int idx = (int) mRandom.GetRange( numBranches );

    for( int i = 0; i < numBranches; i++ )
    {
        if( !node->mBranch[idx].mNode )
            return idx;

        if( ++idx == numBranches )
            idx = 0;
    }

    // This node is fully expanded, so choose the move with highest UCT

    double highestUct = 0;
    int highestIdx = 0;

    for( int i = 0; i < numBranches; i++ )
    {
        double uct = CalculateUct( node, i );
        if( (i == 0) || (uct > highestUct) )
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
    chosenBranch->mVirtualLoss += mOptions->mVirtualLoss;

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
    u64 bestDenom = 0;
    int bestDenomIdx = 0;

    float bestRatio = 0;
    int bestRatioIdx = 0;

    for( int i = 0; i < (int) node->mBranch.size(); i++ )
    {
        if( node->mBranch[i].mScores.mPlays > bestDenom )
        {
            bestDenom = node->mBranch[i].mScores.mPlays;
            bestDenomIdx = i;
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
        printf( "%s%s  %2d) %5s %.15f %.5f %12ld/%-12ld\n", 
            (i == bestRatioIdx)? ">" : " ", 
            (i == bestDenomIdx)? "***" : "   ", 
            i,
            moveText.c_str(), 
            this->CalculateUct( node, i ), 
            node->mBranch[i].mVirtualLoss,
            (u64) node->mBranch[i].mScores.mWins[node->mColor], (u64) node->mBranch[i].mScores.mPlays );
    }
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

    // Remove the virtual loss we added while building the batch

    childInfo.mVirtualLoss -= mOptions->mVirtualLoss;
    if( childInfo.mVirtualLoss < 0 )
        childInfo.mVirtualLoss = 0;

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
        GamePlayer< u64 > player( &batch->mParams, i );
        player.PlayGames( &batch->mPosition[i], &checkScores, 1 );

        assert( checkScores.mWins[0] == batch->mResults[i].mWins[0] );
        assert( checkScores.mWins[1] == batch->mResults[i].mWins[1] );
        assert( checkScores.mPlays == batch->mResults[i].mPlays );
    }
#endif

    for( int i = 0; i < batch->GetCount(); i++ )
        this->DeliverScores( mSearchRoot, batch->mPathFromRoot[i], batch->mResults[i] );
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
    for( auto& worker : mAsyncWorkers )
        worker->Update();
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

        if( mShuttingDown )
            break;

        while( mSearchingNow )
        {
            this->UpdateAsyncWorkers();
            this->ProcessIncomingScores();

            if( mWorkQueue.PeekCount() < mOptions->mMaxPendingJobs )
            {
                auto batch = this->ExpandTree();
                if( batch->GetCount() > 0 )
                    mWorkQueue.Push( batch );

                static int counter = 0;
                if( ++counter > 10 )
                {
                    this->DumpStats( this->mSearchRoot );
                    counter = 0;
                }
            }
            else
            {
                PlatSleep( 1 );
            }

        }
    }
}



