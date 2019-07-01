// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once
#include "IGameState.h"

namespace Chess
{
    #include "Core/Operations.h"
    #include "Core/Defs.h"
    #include "Core/BitBoard.h"
    #include "Core/MoveSpec.h"
    #include "Core/MoveMap.h"
    #include "Core/Position.h"
    #include "Core/MoveList.h"

    class GameState : public IGameState
    {
        Position _Pos;
        MoveMap  _MoveMap;

    public:
        virtual void Reset()
        {
            _Pos.Reset();
            _Pos.CalcMoveMap( &_MoveMap );
        }

        virtual vector< int > FindMoves() const
        {
            MoveList moveList;
            moveList.UnpackMoveMap( _Pos, _MoveMap );

            vector< int > result;
            result.reserve( moveList._Count );

            for( int i = 0; i < moveList._Count; i++ )
                result.push_back( moveList._Move[i].GetAsInt() );

            return result;
        }

        virtual bool MakeMove( int moveToken )
        {
            MoveSpec move;
            move.SetFromInt( moveToken );
            _Pos.Step( move, &_MoveMap );
            return true;
        }

        virtual int GetResult() const
        {
            return (int) _Pos._GameResult;
        }

        virtual string Serialize() const
        {
            char fen[MAX_FEN_LENGTH];
            //PositionToString( _Pos, fen );
            return fen;
        }

        virtual bool Deserialize( const string& fen )
        {
        // if( !StringToPosition( fen.c_str(), &_Pos ) )
                return false;

            _Pos.CalcMoveMap( &_MoveMap );
            return true;
        }
    }
};

