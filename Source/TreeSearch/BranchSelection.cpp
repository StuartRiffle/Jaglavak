// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess/Core.h"
#include "TreeSearch.h"


double TreeSearch::CalculateUct( TreeNode* node, int childIndex )
{
    BranchInfo* nodeInfo    = node->_Info;
    BranchInfo& childInfo   = node->_Branch[childIndex];
    const ScoreCard& scores = childInfo._Scores;

    u64 nodePlays  = MAX( nodeInfo->_Scores._Plays, 1 ); 
    u64 childPlays = MAX( scores._Plays, 1 );
    u64 childWins  = scores._Wins[node->_Color];

    if( _Settings->_DrawsWorthHalf )
    {
        u64 draws = scores._Plays - (scores._Wins[WHITE] + scores._Wins[BLACK]);
        childWins += draws / 2;
    }

    double invChildPlays = 1.0 / childPlays;
    double childWinRatio = childWins * invChildPlays;

    double uct = 
        childWinRatio + 
        sqrt( log( (double) nodePlays ) * 2 * invChildPlays ) * _ExplorationFactor +
        childInfo._VirtualLoss +
        childInfo._Prior;

    return uct;
}

int TreeSearch::GetRandomUnexploredBranch( TreeNode* node )
{
    int numBranches = (int) node->_Branch.size();
    int idx = (int) _RandomGen.GetRange( numBranches );

    for( int i = 0; i < numBranches; i++ )
    {
        if( !node->_Branch[idx]._Node )
            return idx;

        idx = (idx + 1) % numBranches;
    }

    return( -1 );
}

int TreeSearch::SelectNextBranch( TreeNode* node )
{
    int numBranches = (int) node->_Branch.size();
    assert( numBranches > 0 );

    int randomBranch = GetRandomUnexploredBranch( node );
    return randomBranch;

    // This node is fully expanded, so choose the move with highest UCT

    double highestUct = DBL_MIN;
    int highestIdx = 0;

    for( int i = 0; i < numBranches; i++ )
    {
        double uct = CalculateUct( node, i );
        if( uct > highestUct )
        {
            highestUct = uct;
            highestIdx = i;
        }
    }

    return highestIdx;
}


