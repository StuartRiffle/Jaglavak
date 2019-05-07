// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "TreeSearch.h"

TreeSearch::TreeSearch( GlobalOptions* options, u64 randomSeed ) : 
    mOptions( options )
{
    mSearchRoot    = NULL;
    mShuttingDown  = false;
    mSearchRunning = false;
    mRandom.SetSeed( randomSeed );

    mNodePoolEntries = mOptions->mMaxTreeNodes;
    mNodePool = new TreeNode[mNodePoolEntries];

    for( int i = 0; i < mNodePoolEntries; i++ )
    {
        mNodePool[i].mPrev = &mNodePool[i - 1];
        mNodePool[i].mNext = &mNodePool[i + 1];
        mNodePool[i].mCounter = -i;
    }

    mNodePool[0].mPrev = (TreeNode*) &mMruListHead;
    mMruListHead.mNext = &mNodePool[0];

    mNodePool[mNodePoolEntries - 1].mNext = (TreeNode*) &mMruListHead;
    mMruListHead.mPrev = &mNodePool[mNodePoolEntries - 1];
}

~TreeSearch::TreeSearch()
{
    this->StopSearching();

    mShuttingDown = true;
    mSearchThreadGo.Post();
    mSearchThread->join();

    delete mSearchThread;
    delete mNodePool;
}

void TreeSearch::Init()
{
    for( int i = 0; i < mOptions->mNumLocalWorkers; i++ )
    {
        auto worker = new LocalWorker( mOptions, &mWorkQueue, &mDoneQueue );
        mAsyncWorkers.push_back( RC< AsyncWorker >( worker ) );
    }

    if( mOptions->mAllowCuda )
    {
        for( int i = 0; i < CudaWorker::GetDeviceCount(); i++ )
        {
            RC< AsyncWorker > worker( new CudaWorker( mOptions, &mWorkQueue, &mDoneQueue ) );
            worker->Initialize( i, mOptions->mCudaQueueDepth );

            mAsyncWorkers.push_back( worker );
        }
    }

    mSearchThread  = new thread( [this] { this->SearchThread(); } );
    mResultThread  = new thread( [this] { this->ResultThread(); } );
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

    mSearchRunning = true;
    mSearchThreadGo.Post();
}

void TreeSearch::StopSearching()
{
    if( mSearchRunning )
    {
        mSearchRunning = false;
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

void TreeSearch::SetPosition( const Position& startPos, const MoveList* moveList = NULL )
{
    // TODO: recognize position and don't terf the whole tree

    Position pos = startPos;

    if( moveList )
        for( int i = 0; i < moveList->mCount; i++ )
            pos.Step( moveList->mMove[i] );

    if( mSearchRoot )
        mSearchRoot->mInfo = NULL;

    mSearchRoot = AllocNode();
    mSearchRoot->InitPosition( pos );

    mSearchRoot->mInfo = &mRootInfo;
    mRootInfo.mNode = mSearchRoot;
}

void TreeSearch::DebugVerifyMruList()
{
    TreeNode* node = mMruListHead.mNext;
    int count = 0;

    while( node != (TreeNode*) &mMruListHead )
    {
        count++;
        node = node->mNext;
    }

    assert( count == mNodePoolEntries );
}


Position VecToPosition( const vector< float >& tensor )
{
    Position pos;

}

void TreeSearch::CalculatePriorsAsync( TreeNode* node, const MoveList& pathFromRoot )
{
    int numBranches = (int) node->mBranch.size();

    JobRef job( new InferenceJob() );

    job->mGraphName = "CalcPriors";
    job->mCheckpoint = "";
    job->mPathFromRoot = pathFromRoot;
    job->mInputs = PositionToVec( node->mPos );

    mInferenceJobs.Push( job );
}

float TreeSearch::CalculateUct( TreeNode* node, int childIndex )
{
    BranchInfo* nodeInfo    = node->mInfo;
    BranchInfo& childInfo   = node->mBranch[childIndex];
    const ScoreCard& scores = childInfo.mScores;

    u64 nodePlays  = Min( nodeInfo->mScores.mPlays, 1 );
    u64 childPlays = Min( scores.mPlays, 1 );
    u64 childWins  = scores.mWins[node->mColor];
    
    if( mOptions->mDrawsHaveValue )
    {
        int draws = scores.mPlays - (scores.mWins[WHITE] + scores.mWins[BLACK]);
        childWins += draws / 2;
    }

    float invChildPlays = 1.0f / childPlays;
    float childWinRatio = childWins * invChildPlays;
    float exploringness = mOptions->mExplorationFactor * 0.01f;

    float uct = 
        childWinRatio + 
        exploringness * sqrtf( logf( nodePlays ) * 2 * invChildPlays ) +
        childInfo.mPrior;

    return uct;
}

int TreeSearch::SelectNextBranch( TreeNode* node )
{
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

    float highestUct = 0;
    int highestIdx = 0;

    for( int i = 0; i < numBranches; i++ )
    {
        float uct = CalculateUct( node, i );
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

    pathFromRoot.Append( chosenBranch->mMove );

    if( !chosenBranch->mNode )
    {
        // This is a leaf, so create a new node 

        TreeNode* newNode = AllocNode();
        assert( newNode != node );

        MoveToFront( node );

        MoveMap newMap;
        Position newPos = node->mPos;
        newPos.Step( chosenBranch->mMove, &newMap );

        //chosenBranch->mNode = NULL;
        assert( chosenBranch->mNode == NULL );

        newNode->InitPosition( newPos, chosenBranch, newMap ); 
        this->CalculatePriors( newNode, pathFromRoot );

        chosenBranch->mNode = newNode;

        if( newNode->mGameOver )
        {
            newNode->mInfo->mScores = newNode->mGameResult;
            return( newNode->mGameResult );
        }

        batch->Add( newPos, pathFromRoot );

        // Pretend that we played a game here so that the UCT value changes

        ScoreCard scores;
        scores.mPlays = 1;
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

    printf( "Queue length %d\n", mWorkQueue.GetCount() );
    for( int i = 0; i < (int) node->mBranch.size(); i++ )
    {
        string moveText = SerializeMoveSpec( node->mBranch[i].mMove );
        printf( "%s%s  %2d) %5s %.15f %12ld/%-12ld\n", 
            (i == bestRatioIdx)? ">" : " ", 
            (i == bestDenomIdx)? "***" : "   ", 
            i,
            moveText.c_str(), 
            this->CalculateUct( node, i ), 
            node->mBranch[i].mScores.mWins[node->mColor], node->mBranch[i].mScores.mPlays );
    }
}


void TreeSearch::DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& score, int depth = 0 )
{
    if( depth >= result.mPathFromRoot.mCount )
        return;

    MoveSpec move = result.mPathFromRoot.mMove[depth];

    int childIdx = node->FindMoveIndex( move );
    if( childIdx < 0 )
        return;

    TreeNode* child = node->mBranch[childIdx].mNode;
    if( child == NULL )
        return;

    DeliverScores( child, result, depth + 1 );

    node->mBranch[childIdx].mScores += result.mScores;
}

void TreeSearch::ProcessScoreBatch( BatchRef& batch )
{
    for( int i = 0; i < batch->mCount; i++ )
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

BatchRef TreeSearch::ExpandTree()
{
    BatchRef batch( new PlayoutBatch );

    int batchSize = Min( mOptions->mBatchSize, PLAYOUT_BATCH_MAX );
    for( int i = 0; i < batchSize; i++ )
    {
        MoveList pathFromRoot;
        ScoreCard rootScores = this->ExpandAtLeaf( pathFromRoot, mSearchRoot, batch );

        mSearchRoot->mInfo->mScores += rootScores;
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

        while( mSearchNow )
        {
            this->UpdateAsyncWorkers();
            this->ProcessIncomingScores();
            this->ProcessIncomingInference();

            auto batch = this->ExpandTree();
            mWorkQueue->Push( batch );
        }
    }
}



