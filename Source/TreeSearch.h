// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeSearch
{
    TreeNode*               mNodePool;
    size_t                  mNodePoolEntries;
    TreeLink                mMruListHead;
    TreeNode*               mSearchRoot;
    BranchInfo              mRootInfo;
    UciSearchConfig         mUciConfig;
    GlobalOptions*          mOptions;
    std::thread*            mSearchThread;
    Semaphore               mSearchThreadActive;
    Semaphore               mSearchThreadIdle;
    volatile bool           mShuttingDown;
    volatile bool           mSearchRunning;
    RandomGen               mRandom;
    BatchQueue         mPendingQueue;
    BatchQueue      mDoneQueue;

    std::vector< std::shared_ptr< IAsyncWorker > > mAsyncWorkers;

    TreeNode* AllocNode();
    void MoveToFront( TreeNode* node );

    float CalculateUct( TreeNode* node, int childIndex )
    int SelectNextBranch( TreeNode* node )
    ScoreCard ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node );
    void ExpandAtLeaf();
    void DumpStats( TreeNode* node );

    void ProcessResult( TreeNode* node, const PlayoutResultRef& result, int depth = 0 );
    void ProcessAsyncResults();
    void UpdateAsyncWorkers();
    void SearchThread();

public:
    TreeSearch( GlobalOptions* options, u64 randomSeed = 1 );
    ~TreeSearch();

    void Init();
    void Reset();
    void SetPosition( const Position& pos );
    void StartSearching( const UciSearchConfig& config );
    void StopSearching();
};




