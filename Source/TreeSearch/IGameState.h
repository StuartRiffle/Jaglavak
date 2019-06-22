// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

class IGameState
{
public:
    virtual void Reset() = 0;
    virtual vector< int > FindMoves() const = 0;
    virtual void MakeMove( int token ) = 0;
    virtual int GetResult() const = 0;

    virtual string Serialize() const = 0;
    virtual bool Deserialize( const string& str ) = 0;
};

class ChessGameState : public IGameState
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

    virtual void MakeMove( int moveToken )
    {
        MoveSpec move;
        move.SetFromInt( moveToken );
        _Pos.Step( move, &_MoveMap );
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
};
