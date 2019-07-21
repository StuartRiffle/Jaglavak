// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    bool _Done = false;
    u64 _YieldCounter = 0;
    u64 _TickQueued = 0;

    // Inputs
    PlayoutParams       _Params;
    vector< Position >  _Position;

    // Outputs
    vector< ScoreCard > _GameResults;

    int GetCount() const { return (int) _Position.size(); }
};

typedef shared_ptr< PlayoutBatch >  BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

