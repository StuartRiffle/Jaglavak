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

/*
class IGameState
{
public:
    virtual void Reset() = 0;
    virtual vector< int > FindMoves() const = 0;
    virtual void MakeMove( int token ) = 0;
    virtual GameResult GetResult() const = 0;

    virtual string Serialize() const = 0;
    virtual bool Deserialize( const string& str ) = 0;
};

class ChessGameState : public IGameState
{
    Position _Pos;
    MoveMap  _MoveMap;

public:
    virtual void Reset()
    {
        _Pos.Reset();
        _Pos.CalcMoveMap( &_MoveMap );
    }

    virtual vector< int > FindMoves() const
    {
        MoveList moveList;
        moveList.UnpackMoveMap( _Pos, _MoveMap );

        vector< int > result;
        result.reserve( moveList._Count );

        for( int i = 0; i < moveList._Count; i++ )
            result.push_back( moveList_Move[i].GetAsInt() );

        return result;
    }

    virtual void MakeMove( int moveToken )
    {
        MoveSpec move;
        move.SetFromInt( moveToken );
        _Position.Step( move, &_MoveMap );
    }

    virtual int GetResult() const
    {
        return _Pos._GameResult;
    }

    virtual string Serialize() const
    {
        char fen[MAX_FEN_LENGTH];
        PositionToString( _Pos, fen );
        return fen;
    }

    virtual bool Deserialize( const string& fen )
    {
        if( !StringToPosition( fen.c_str(), &_Pos ) )
            return false;

        _Pos.CalcMoveMap( &_MoveMap );
        return true;
    }
};
*/

class TreeSearch
{
    GlobalSettings*         _Settings = NULL;
    RandomGen               _RandomGen;
    BatchQueue              _BatchQueue;
    BatchRef                _Batch;

    MoveList                _GameHistory;
    PlayoutParams           _PlayoutParams;

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

    bool                    _DrawsWorthHalf = false;
    float                   _ExplorationFactor = 0;

    typedef shared_ptr< AsyncWorker > AsyncWorkerRef;
    vector< AsyncWorkerRef > _Workers;

    void SearchThread();
    void SearchFiber();
    void FlushBatch();

    // BranchSelection.cpp

    double CalculateUct( TreeNode* node, int childIndex );
    int GetRandomUnexploredBranch( TreeNode* node );
    int SelectNextBranch( TreeNode* node );

    // Expansion.cpp

    ScoreCard ExpandAtLeaf( TreeNode* node, int depth = 1 );

    // TimeControl.cpp

    bool IsTimeToMove();

    // UciStatus.cpp

    void ExtractBestLine( TreeNode* node, MoveList* dest );
    int EstimatePawnAdvantageForMove( const MoveSpec& spec );
    MoveSpec SendUciStatus();
    void SendUciBestMove();

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




