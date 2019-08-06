// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"
#include "FEN.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost;
namespace pt = property_tree;

#include "RpcClient.h"

void TreeSearch::EstimatePriors( TreeNode* node )
{
    CallRef call( new RpcCall() );

    call->_ServerType = "inference";
    call->_Inputs.put( "type", "estimate_priors" );
    call->_Inputs.put( "position", SerializePosition( node->_Pos ) );

    _RpcClient->Call( call );
    while( !call->_Done )
    {
        // ------------------------
        _SearchFibers.YieldFiber();
        // ------------------------
    }

    if( call->_Success )
    {
        vector< string > moveList  = SplitString( call->_Outputs.get< string >( "moves" ) );
        vector< string > valueList = SplitString( call->_Outputs.get< string >( "values" ) );
        assert( moveList.size() == valueList.size() );
        assert( moveList.size() == node->_Branch.size() );

        map< string, float > moveValue;
        for( int i = 0; i < (int) moveList.size(); i++ )
            moveValue[moveList[i]] = (float) atof( valueList[i].c_str() );

        for( int i = 0; i < node->_Branch.size(); i++ )
            node->_Branch[i]._Prior = moveValue[SerializeMoveSpec( node->_Branch[i]._Move )];
    }

    float scaleNoise = _Settings->Get< float >( "Search.PriorNoise" );
    for( int i = 0; i < node->_Branch.size(); i++ )
    {
        float noise = _RandomGen.GetNormal() * scaleNoise;
        node->_Branch[i]._Prior += noise;
    }
}

