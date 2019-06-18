// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

bool TreeSearch::IsTimeToMove()
{
    bool    whiteToMove     = _SearchRoot->_Pos._WhiteToMove; 
    int     requiredMoves   = _UciConfig._TimeControlMoves;
    int     timeBuffer      = _Options->_TimeSafetyBuffer;
    int     timeElapsed     = _SearchTimer.GetElapsedMs() + timeBuffer;
    int     timeInc         = whiteToMove? _UciConfig._WhiteTimeInc  : _UciConfig._BlackTimeInc;
    int     timeLeftAtStart = whiteToMove? _UciConfig._WhiteTimeLeft : _UciConfig._BlackTimeLeft;
    int     timeLimit       = _UciConfig._TimeLimit;
    int     timeLeft        = timeLeftAtStart - timeElapsed;

    if( timeLimit )
        if( timeElapsed >= timeLimit )
            return true;

    if( requiredMoves && timeLeftAtStart )
        if( timeElapsed >= (timeLeftAtStart / requiredMoves) )
            return true;

    if( _UciConfig._NodesLimit )
        if( _Metrics._NumNodesCreated >= _UciConfig._NodesLimit )
            return true;

    if( _UciConfig._DepthLimit )
        if( _DeepestLevelSearched > _UciConfig._DepthLimit )
            return true;

    return false;
}

