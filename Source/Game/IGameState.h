// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

enum
{
    RESULT_UNKNOWN,
    RESULT_WHITE_WIN,
    RESULT_BLACK_WIN,
    RESULT_DRAW
};

class IGameState
{
public:
    typedef int MoveType;

    virtual void Reset() = 0;
    virtual vector< MoveType > FindMoves() const = 0;
    virtual bool MakeMove( MoveType move, IGameState* dest ) = 0;
    virtual int GetResult() const = 0;

    virtual string Serialize() const = 0;
    virtual bool Deserialize( const string& str ) = 0;
};

template< typename GAMESTATE >
class IGameplayProvider
{
public:
    virtual void Init( const PlayoutParams* params, u64 randomSalt ) = 0;
    virtual void PlayGamesOut( const GAMESTATE* pos, ScoreCard* outScores, int count ) = 0;
    virtual void PlayOneGame( const GAMESTATE& startPos, ScoreCard* outScores ) = 0;
};

class GameplayProviderBase
{
    const PlayoutParams* _Params;
    RandomGen _RandomGen;

public:
    void Init( const PlayoutParams* params, u64 randomSalt )
    {
        _Params = params;
        _RandomGen.SetSeed( params->_RandomSeed ^ Mix64( randomSalt ) );
    }

    virtual void PlayGamesOut( const GAMESTATE* startPos, ScoreCard* outScores, int count )
    {
        assert( (uintptr_t) pos % sizeof( SIMD ) == 0 );

        int totalIters = simdCount * _Params->_NumGamesEach;

        #pragma omp parallel for schedule(dynamic) if (_Params->_Multicore)
        for( int i = 0; i < totalIters; i++ )
        {
            int idx = i % count;
            this->PlayOneGame( pos + idx, outScores + idx );
        }
    }

    virtual void PlayOneGame( const GAMESTATEI* startPos, ScoreCard* outScores ) = 0;

};


