// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct TreeNode;

struct BranchInfo
{
    TreeNode*   mNode;
    MoveSpec    mMove;
    ScoreCard   mScores;
    float       mPrior;

#if DEBUG    
    char        mMoveText[MAX_MOVETEXT_LENGTH];
#endif

    BranchInfo() { memset( this, 0, sizeof( this ) ); }
};

struct ALIGN_SIMD TreeLink
{
    TreeNode*           mPrev;
    TreeNode*           mNext;
};

struct TreeNode : public TreeLink
{
    Position            mPos;
    BranchInfo*         mInfo;
    int                 mColor;
    std::vector<BranchInfo>  mBranch;

    bool                mGameOver;
    ScoreCard           mGameResult;

    TreeNode() : mInfo( NULL ), mGameOver( false ), mCounter( -1 ), mTouch( 0 ) {}
    ~TreeNode() { Clear(); }

    void Init( const Position& pos, BranchInfo* info = NULL );
    void Clear();
    int FindMoveIndex( const MoveSpec& move );
    void SanityCheck();
};

struct TreeSearcher
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
    TreeSearcher( GlobalOptions* options, u64 randomSeed = 1 );
    ~TreeSearcher();

    void Init();
    void Reset();
    void SetPosition( const Position& pos );
    void StartSearching( const UciSearchConfig& config );
    void StopSearching();
};




