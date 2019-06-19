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
    GlobalOptions*          _Options = NULL;
    RandomGen               _RandomGen;
    BatchQueue              _BatchQueue;
    BatchRef                _Batch;

    Position                _Position;
    MoveList                _GameHistory;

    UciSearchConfig         _UciConfig = {};
    unique_ptr< SearchTree > _SearchTree;
    unique_ptr< thread >    _SearchThread;
    volatile bool           _SearchExit = false;
    Timer                   _SearchTimer;

    Timer                   _UciUpdateTimer;
    int                     _DeepestLevelSearched = 0;
    TreeSearchMetrics       _Metrics;
    TreeSearchMetrics       _SearchStartMetrics;
    TreeSearchMetrics       _StatsStartMetrics;

    typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
    vector< AsyncWorkerRef > _Workers;

    void SearchThread();
    void SearchFiber();
    void FlushBatch();

    // BranchSelection.cpp

    double CalculateUct( TreeNode* node, int childIndex );
    int GetRandomUnexploredBranch( TreeNode* node );
    int SelectNextBranch( TreeNode* node );

    // PriorEstimation.cpp

    void EstimatePriors( TreeNode* node );

    // Expansion.cpp

    ScoreCard ExpandAtLeaf( TreeNode* node );

    // TimeControl.cpp

    bool IsTimeToMove();

    // UciStatus.cpp

    void ExtractBestLine( TreeNode* node, MoveList* dest );
    int EstimatePawnAdvantageForMove( const MoveSpec& spec );
    MoveSpec SendUciStatus();

public:

    TreeSearch( GlobalOptions* options );
    ~TreeSearch();

    void Init();
    void Reset();
    void SetPosition( const Position& pos, const MoveList* moveList = NULL );
    void StartSearching();
    void StopSearching();

    void SetUciSearchConfig( const UciSearchConfig& config ) { _UciConfig = config; }
};




