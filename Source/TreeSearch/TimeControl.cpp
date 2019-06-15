// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle

#include "Platform.h"
#include "Chess.h"
#include "Common.h"

bool TreeSearch::IsTimeToMove()
{
    const float MS_TO_SEC = 0.001f;

    bool    whiteToMove     = _SearchRoot->_Pos._WhiteToMove; 
    int     requiredMoves   = _UciConfig._TimeControlMoves;
    float   timeBuffer      = _Options->_TimeSafetyBuffer * MS_TO_SEC;
    float   timeElapsed     = _SearchTimer.GetElapsedSec() + timeBuffer;
    float   timeInc         = (whiteToMove? _UciConfig._WhiteTimeInc  : _UciConfig._BlackTimeInc)  * MS_TO_SEC;
    float   timeLeftAtStart = (whiteToMove? _UciConfig._WhiteTimeLeft : _UciConfig._BlackTimeLeft) * MS_TO_SEC;
    float   timeLimit       = _UciConfig._TimeLimit * MS_TO_SEC;
    float   timeLeft        = timeLeftAtStart - timeElapsed;

    if( timeLimit )
        if( timeElapsed > timeLimit )
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


