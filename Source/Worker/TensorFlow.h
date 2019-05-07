// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class TensorFlowSession
{
    string mGraphName;
    string mCheckpointName;
    bool mReady;

    tensorflow::Session* mSession;
    tensorflow::SessionOptions mSessionOptions;
    tensorflow::MetaGraphDef mGraphDef;

    typedef pair<string, tensorflow::Tensor> NamedInput;
    typedef vector< NamedInput > InputDict;

    tensorFlow::TensorShape mInputShape;
    InputDict mInputDict;

    TensorFlowSession( const char* graphName, const char* checkpointName = "" ) :
        mGraphName( graphName ),
        mCheckpointName( checkpointName ),
        mSession( NULL ),
        mSessionOptions(),
        mReady( false )
    {
        tensorflow::status status = tensorflow::NewSession( mSessionOptions, mSession );
        if( status.ok() )
        {
            status = tensorflow::ReadBinaryProto( Env::Default(), mGraphName, &mGraphDef );
            if( status.ok() )
            {
                status = mSession->CreateGraph( mGraphDef );
                if( status.ok() )
                {
                    mReady = true;
                }
            }
        }
    }

    ~TensorFlowSession()
    {
        if( mSession )
            mSession->Close();
    }

    bool Run( const vector< float >& inputs, vector< float >& outputs )
    {
        if( !mReady )
            return false;

        InputDict graphInputs = { "input", inputs.data() };
        vector< tensorflow::Tensor > graphOutputs;

        mSession->Run( 
            inputDict, )

        if( graphOutputs.size() != 1 )
            return false;

        auto& resultTensor = graphOutputs[0];
        auto flatArray = resultTensor.flat< float >();
        auto dims = flatArray.dimensions();
        int count = (int) dims[0];

        outputs.resize( count );
        for( int i = 0; i < count; i++ )
            outputs[i] = flatArray( i );

        return true;
    }
};
