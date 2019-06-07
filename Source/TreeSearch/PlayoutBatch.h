// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    // Inputs
    vector< Position >  _Position;
    PlayoutParams       _Params;

    // Outputs
    vector< ScoreCard > _GameResults;

    // These gets carried along so we know where the results should go
    vector< MoveList >  _PathFromRoot;

    int GetCount() const
    {
        return (int) _Position.size();
    }

    void Append( const Position& pos, const MoveList& pathFromRoot )
    {
        _Position.push_back( pos );
        _PathFromRoot.push_back( pathFromRoot );

        assert( _Position.size() == _PathFromRoot.size() );
    }
};

typedef shared_ptr< PlayoutBatch >  BatchRef;
typedef vector< BatchRef >          BatchVec;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

