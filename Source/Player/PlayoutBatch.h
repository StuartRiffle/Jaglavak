// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

struct PlayoutBatch
{
    // Inputs

    PlayoutParams mParams;
    vector< Position > mPosition;

    // Outputs

    vector< ScoreCard > mResults;

    // This gets carried along so we know where the results should go

    vector< MoveList > mPathFromRoot;

    PlayoutBatch()
    {
        memset( &mParams, 0, sizeof( mParams ) );
    }

    int GetCount() const
    {
        return (int) mPosition.size();
    }

    void Append( const Position& pos, const MoveList& pathFromRoot )
    {
        mPosition.push_back( pos );
        mPathFromRoot.push_back( pathFromRoot );

        assert( mPosition.size() == mPathFromRoot.size() );
    }
};

#if !ON_CUDA_DEVICE
typedef RC< PlayoutBatch > BatchRef;
typedef ThreadSafeQueue< BatchRef > BatchQueue;
#endif

