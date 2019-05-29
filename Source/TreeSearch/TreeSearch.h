// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct UciSearchConfig
{
    int         mWhiteTimeLeft;   
    int         mBlackTimeLeft;   
    int         mWhiteTimeInc;    
    int         mBlackTimeInc;    
    int         mTimeControlMoves;
    int         mMateSearchDepth; 
    int         mDepthLimit;       
    int         mNodesLimit;       
    int         mTimeLimit; 
    MoveList    mLimitMoves;

    UciSearchConfig()   { this->Clear(); }
    void Clear()        { memset( this, 0, sizeof( *this ) ); }
};

struct TreeSearchParameters
{
    int         mBatchSize;
    int         mMaxPending;
    int         mInitialPlayouts;
    int         mAsyncPlayouts;
};

struct TreeSearchMetrics
{
    u64         mNumBatchesMade;
    u64         mNumBatchesDone;
    u64         mNumNodesCreated;
    u64         mNumGamesPlayed;

    void Clear() { memset( this, 0, sizeof( *this )); }

    void operator+=( const TreeSearchMetrics& rhs )
    {
        u64* src = (u64*) &rhs;
        u64* dest = (u64*) this;
        int count = (int) (sizeof( *this ) / sizeof( u64 ));

        for( int i = 0; i < count; i++ )
            dest[i] += src[i];
    }
};

struct TreeSearch
{
    GlobalOptions*          mOptions;
    UciSearchConfig         mUciConfig;
    TreeSearchParameters    mSearchParams;
    RandomGen               mRandom;

    TreeSearchMetrics       mMetrics;
    TreeSearchMetrics       mSearchStartMetrics;
    TreeSearchMetrics       mStatsStartMetrics;

    uint8_t*                mNodePoolBuf;
    TreeNode*               mNodePool;
    size_t                  mNodePoolEntries;
    TreeLink                mMruListHead;

    TreeNode*               mSearchRoot;
    BranchInfo              mRootInfo;

    unique_ptr< thread >    mSearchThread;
    Semaphore               mSearchThreadGo;
    Semaphore               mSearchThreadIsIdle;
    volatile bool           mSearchingNow;
    volatile bool           mShuttingDown;
    Timer                   mSearchTimer;
    int                     mDeepestLevelSearched;
    Timer                   mUciUpdateTimer;

    BatchQueue              mWorkQueue;
    BatchQueue              mDoneQueue;

    typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
    vector< AsyncWorkerRef > mAsyncWorkers;

    TreeNode* AllocNode();
    void MoveToFront( TreeNode* node );

    double CalculateUct( TreeNode* node, int childIndex );
    void CalculatePriors( TreeNode* node, MoveList& pathFromRoot );
    int SelectNextBranch( TreeNode* node );
    ScoreCard ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node, BatchRef batch );

    void DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& score, int depth = 0 );
    void ProcessScoreBatch( BatchRef& batch )    ;

    BatchRef CreateNewBatch();
    void DumpStats( TreeNode* node );

    bool IsTimeToMove();
    void ProcessIncomingScores();
    void UpdateAsyncWorkers();
    void AdjustForWarmup();
    void SearchThread();

    int GetRandomUnexploredBranch( TreeNode* node );
    MoveSpec SendUciStatus();
    void ExtractBestLine( TreeNode* node, MoveList* dest );


public:

    TreeSearch( GlobalOptions* options, u64 randomSeed = 1 );
    ~TreeSearch();

    void Init();
    void Reset();
    void SetPosition( const Position& pos, const MoveList* moveList = NULL );
    void SetUciSearchConfig( const UciSearchConfig& config );
    void StartSearching();
    void StopSearching();
};




