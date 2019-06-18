// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct UciSearchConfig
{
    int         _WhiteTimeLeft;   
    int         _BlackTimeLeft;   
    int         _WhiteTimeInc;    
    int         _BlackTimeInc;    
    int         _TimeControlMoves;
    int         _MateSearchDepth; 
    int         _DepthLimit;       
    int         _NodesLimit;       
    int         _TimeLimit; 
    MoveList    _LimitMoves;

    UciSearchConfig()   { this->Clear(); }
    void Clear()        { memset( this, 0, sizeof( *this ) ); }
};

struct TreeSearchMetrics
{
    u64         _NumBatchesMade;
    u64         _NumBatchesDone;
    u64         _NumNodesCreated;
    u64         _NumGamesPlayed;

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


class TreeSearch
{
    GlobalOptions*          _Options;
    RandomGen               _Random;
    unique_ptr< Tree >      _Tree;

    MoveList                _GameHistory;

    BatchQueue              _BatchQueue;
    BatchRef                _Batch;

    unique_ptr< thread >    _SearchThread;
    volatile bool           _SearchExit;

    Timer                   _SearchTimer;
    Timer                   _UciUpdateTimer;
    int                     _DeepestLevelSearched;
    TreeSearchMetrics       _Metrics;
    TreeSearchMetrics       _SearchStartMetrics;
    TreeSearchMetrics       _StatsStartMetrics;

    typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
    vector< AsyncWorkerRef > _AsyncWorkers;

    TreeNode* AllocNode();
    void MoveToFront( TreeNode* node );

    double CalculateUct( TreeNode* node, int childIndex );
    void CalculatePriors( TreeNode* node, MoveList& pathFromRoot );
    int SelectNextBranch( TreeNode* node );
    ScoreCard PlayBranchGames( MoveList& pathFromRoot, TreeNode* node, BatchRef batch );
    void CreateNewNode( MoveList& pathFromRoot, TreeNode* node, int branchIdx );
    ScoreCard ExpandAtLeaf( MoveList& pathFromRoot, TreeNode* node, BatchRef batch );

    void DeliverScores( TreeNode* node, MoveList& pathFromRoot, const ScoreCard& score, int depth = 0 );
    void ProcessScoreBatch( BatchRef& batch );

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
    int EstimatePawnAdvantageForMove( const MoveSpec& spec );

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




