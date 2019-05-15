// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct UciSearchConfig
{
    int                 mWhiteTimeLeft;   
    int                 mBlackTimeLeft;   
    int                 mWhiteTimeInc;    
    int                 mBlackTimeInc;    
    int                 mTimeControlMoves;
    int                 mMateSearchDepth; 
    int                 mDepthLimit;       
    int                 mNodesLimit;       
    int                 mTimeLimit; 
    MoveList            mLimitMoves;

    UciSearchConfig()   { this->Clear(); }
    void Clear()        { memset( this, 0, sizeof( *this ) ); }
};

struct TreeSearch
{
    GlobalOptions*          mOptions;
    UciSearchConfig         mUciConfig;
    RandomGen               mRandom;

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

    PlayoutParams GetPlayoutParams();
    BatchRef ExpandTree();
    void DumpStats( TreeNode* node );

    void ProcessIncomingScores();
    void UpdateAsyncWorkers();
    void SearchThread();

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




