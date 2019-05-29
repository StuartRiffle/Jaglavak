// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    // Inputs
    vector< Position > mPosition;
    PlayoutParams mParams;

    // Outputs
    vector< ScoreCard > mResults;

    // This gets carried along so we know where the results should go
    vector< MoveList > mPathFromRoot;

    int GetCount() const
    {
        return (int) mPosition.size();
    }

    void Append( const PlayoutParams& params, const Position& pos, const MoveList& pathFromRoot )
    {
        mPosition.push_back( pos );
        mPathFromRoot.push_back( pathFromRoot );

        assert( mPosition.size() == mPathFromRoot.size() );
    }
};

typedef shared_ptr< PlayoutBatch >  BatchRef;
typedef vector< BatchRef >          BatchVec;
typedef ThreadSafeQueue< BatchRef > BatchQueue;

