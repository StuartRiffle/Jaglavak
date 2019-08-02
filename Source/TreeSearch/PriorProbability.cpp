// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Jaglavak.h"
#include "TreeSearch.h"
#include "FEN.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace boost;
namespace pt = property_tree;

#include "RpcClient.h"

/*
struct EncodedPosition
{
    float   _Pawns[64];
    float   _Knights[64];
    float   _Bishops[64];
    float   _Rooks[64];
    float   _Queens[64];
    float   _Kings[64];
    float   _SideToPlay;

    void Encode( const Position& pos )
    {
        for( int i = 0; i < 64; i++ )
        {
            u64 bit = 1ULL << i;

            _Pawns[i]   =  (pos._WhitePawns   & bit)? 1 : ((pos._BlackPawns   & bit)? -1 : 0);
            _Knights[i] =  (pos._WhiteKnights & bit)? 1 : ((pos._BlackKnights & bit)? -1 : 0);
            _Bishops[i] =  (pos._WhiteBishops & bit)? 1 : ((pos._BlackBishops & bit)? -1 : 0);
            _Rooks[i]   =  (pos._WhiteRooks   & bit)? 1 : ((pos._BlackRooks   & bit)? -1 : 0);
            _Queens[i]  =  (pos._WhiteQueens  & bit)? 1 : ((pos._BlackQueens  & bit)? -1 : 0);
            _Kings[i]   =  (pos._WhiteKing    & bit)? 1 : ((pos._BlackKing    & bit)? -1 : 0);
        }

        _SideToPlay = pos._WhiteToMove ? 1 : -1;
    }
};

struct EncodedMove
{
    float  _SrcDest[64][64];
    float  _Promo[4];

    void Encode( const MoveSpec& move )
    {
        memset( this, 0, sizeof( *this ) );

        _SrcDest[move._Src][move._Dest] = 1;

        _Promo[0] = (move._Promo == PROMOTION_KNIGHT)? 1 : 0;
        _Promo[1] = (move._Promo == PROMOTION_BISHOP)? 1 : 0;
        _Promo[2] = (move._Promo == PROMOTION_ROOK)?   1 : 0;
        _Promo[3] = (move._Promo == PROMOTION_QUEEN)?  1 : 0;
    }

    void ExtractMoveProbability( const MoveList& moveList, float* dest, bool normalize = true ) const
    {
        float totalSquared = 0;
        for( int i = 0; i < moveList._Count; i++ )
        {
            const MoveSpec& move = moveList._Move[i];
            float prob = _SrcDest[move._Src][move._Dest];

            if( move._Promo )
            {
                float highest = MAX( _Promo[0], MAX( _Promo[1], MAX( _Promo[2], _Promo[3] ) ) );

                switch( move._Promo )
                {
                    case PROMOTION_KNIGHT: if( _Promo[0] < highest ) prob = 0; break;
                    case PROMOTION_BISHOP: if( _Promo[1] < highest ) prob = 0; break;
                    case PROMOTION_ROOK:   if( _Promo[2] < highest ) prob = 0; break;
                    case PROMOTION_QUEEN:  if( _Promo[3] < highest ) prob = 0; break;
                }
            }

            dest[i] = prob;
            totalSquared += (prob * prob);
        }

        if( normalize )
        {
            float scale = sqrtf( totalSquared );
            for( int i = 0; i < moveList._Count; i++ )
                dest[i] *= scale;
        }
    }
};

*/

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

