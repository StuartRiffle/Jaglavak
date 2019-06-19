// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    bool _Done = false;
    u64 _YieldCounter = 0;

    // Inputs
    PlayoutParams       _Params;
    vector< Position >  _Position;

    // Outputs
    vector< ScoreCard > _GameResults;


};

typedef shared_ptr< PlayoutBatch >  BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

