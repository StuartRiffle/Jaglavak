// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    // Inputs
    vector< PlayoutRequest > mRequests;

    // Outputs
    vector< PlayoutResult > mResults;

    // This gets carried along so we know where the results should go
    vector< MoveList > mPathFromRoot;

    int GetCount() const
    {
        return (int) mRequests.size();
    }

    void Append( const PlayoutParams& params, const Position& pos, const MoveList& pathFromRoot )
    {
        PlayoutRequest req;
        req.mParams = params;
        req.mPosition = pos;

        mRequests.push_back( req );
        mPathFromRoot.push_back( pathFromRoot );

        assert( mRequests.size() == mPathFromRoot.size() );
    }
};

typedef shared_ptr< PlayoutBatch > BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

