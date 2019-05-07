// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

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

    PTR< thread >            mSearchThread;
    Semaphore               mSearchThreadShouldGo;
    Semaphore               mSearchThreadIsStopped;
    volatile bool           mSearchRunning;
    volatile bool           mShuttingDown;

    BatchQueue              mWorkQueue;
    BatchQueue              mDoneQueue;

    vector< AsyncWorkerRef > mAsyncWorkers;

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




