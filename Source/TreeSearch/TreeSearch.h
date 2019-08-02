// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

#include "SearchTree.h"

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

class RpcClient;

class TreeSearch
{
    GlobalSettings*         _Settings = NULL;
    RandomGen               _RandomGen;
    BatchQueue              _BatchQueue;
    BatchRef                _Batch;

    Position                _GameStartPosition;
    MoveList                _GameHistory;
    PlayoutParams           _PlayoutParams;

    UciSearchConfig         _UciConfig = {};
    unique_ptr< SearchTree > _SearchTree;
    unique_ptr< thread >    _SearchThread;
    unique_ptr< RpcClient > _RpcClient;
    volatile bool           _SearchExit = false;
    Timer                   _SearchTimer;

    Timer                   _UciUpdateTimer;
    int                     _DeepestLevelSearched = 0;
    Metrics                 _Metrics;
    Metrics                 _SearchStartMetrics;
    Metrics                 _StatsStartMetrics;

    bool                    _DrawsWorthHalf = false;
    float                   _ExplorationFactor = 0;

    FiberSet                _SearchFibers;

    typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
    vector< AsyncWorkerRef > _Workers;


    void ___SEARCH_THREAD___();   // These declarations are goofy so that they
    void ___SEARCH_FIBER___();    // are easier to see in a callstack

    void UpdateUciStatus();
    void UpdateFibers();

    // BranchSelection.cpp

    double CalculateUct( TreeNode* node, int childIndex );
    int GetRandomUnexploredBranch( TreeNode* node );
    int SelectNextBranch( TreeNode* node );

    // Expansion.cpp

    ScoreCard ExpandAtLeaf( TreeNode* node, int depth = 1 );
    void FlushBatch();

    // TimeControl.cpp

    bool IsTimeToMove();

    // UciStatus.cpp

    void ExtractBestLine( TreeNode* node, MoveList* dest );
    int EstimatePawnAdvantageForMove( const MoveSpec& spec );
    MoveSpec SendUciStatus();
    void SendUciBestMove();

    // Priors.cpp

    void EstimatePriors( TreeNode* node );

public:

    TreeSearch( GlobalSettings* settings );
    ~TreeSearch();

    void Init();
    void Reset();
    void SetPosition( const Position& pos, const MoveList* moveList = NULL );
    Position GetPosition() const;
    void StartSearching();
    void StopSearching();

    void SetUciSearchConfig( const UciSearchConfig& config ) { _UciConfig = config; }
};




