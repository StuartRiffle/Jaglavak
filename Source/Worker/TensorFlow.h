// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

// WIP

#include "tensorflow/c/c_api.h"

class TensorFlowSession
{
    string _GraphName;
    string _CheckpointName;
    bool _Ready;

    TF_Session* _Session;
    TF_SessionOptions _SessionOptions;
    TF_MetaGraphDef _GraphDef;

    typedef pair<string, TF_Tensor> NamedInput;
    typedef vector< NamedInput > InputDict;

    tensorFlow::TensorShape _InputShape;
    InputDict _InputDict;

    TensorFlowSession( const char* graphName, const char* checkpointName = "" ) :
        _GraphName( graphName ),
        _CheckpointName( checkpointName ),
        _Session( NULL ),
        _SessionOptions(),
        _Ready( false )
    {
        TF_status status = TF_NewSession( _SessionOptions, _Session );
        if( status.ok() )
        {
            status = TF_ReadBinaryProto( Env::Default(), _GraphName, &_GraphDef );
            if( status.ok() )
            {
                status = _Session->CreateGraph( _GraphDef );
                if( status.ok() )
                {
                    _Ready = true;
                }
            }
        }
    }

    ~TensorFlowSession()
    {
        if( _Session )
            _Session->Close();
    }

    bool Run( const vector< float >& inputs, vector< float >& outputs )
    {
        if( !_Ready )
            return false;

        InputDict graphInputs = { "input", inputs.data() };
        vector< TF_Tensor > graphOutputs;

        _Session->Run( 
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
