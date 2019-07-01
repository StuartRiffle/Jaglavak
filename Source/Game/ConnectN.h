// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once
#include "IGameState.h"

class GameStateConnectN : public IGameState
{
    enum
    {
        MAX_ROWS = 16,
        MAX_COLS = 16,
    };

    enum
    {
        RED = 1,
        YELLOW = -1,
    }

    vector< int8_t > _Slot;
    int8_t _Rows;
    int8_t _Cols;
    int8_t _RunLength;
    int8_t _MovesPlayed;
    int8_t _Result;

public:
    void Init( int runLength = 4, int rows = 6, int cols = 7 )
    {
        _Rows = rows;
        _Cols = cols;
        _Slot.resize( _Rows * _Cols );

        _RunLength = runLength;
        _MovesPlayed = 0;
        _Result = RESULT_UNKNOWN;
    }

    virtual void FindMoves( vector< int >& dest ) const
    {
        dest.reserve( _Cols );

        for( int x = 0; x < _Cols; x++ )
            if( PieceAt( _Rows - 1, x ) == 0 )
                moves.push_back( x );
    }

    virtual bool MakeMove( int col )
    {
        if ( _Result != RESULT_UNKNOWN )
            return;

        assert( col < _Cols );
        if( col >= _Cols )
            return false;

        int row;
        for( row = 0; row < _Rows; row++ )
            if( PieceAt( row, col ) == 0 )
                break;

        assert( row < _Rows );
        if( row >= _Rows )
            return false;

        int playerIdx = _MovesPlayed & 1;
        _MovesPlayed++;

        PieceAt( row, col ) = (playerIdx == 0)? 1 : -1;

        bool won =
            (CountRunLength( row, col, 0,  1 ) == _RunLength) ||
            (CountRunLength( row, col, 1,  1 ) == _RunLength) ||
            (CountRunLength( row, col, 1,  0 ) == _RunLength) ||
            (CountRunLength( row, col, 1, -1 ) == _RunLength);

        if( won )
        {
            _Result = (playerIdx == 0)? RESULT_WHITE_WIN : RESULT_BLACK_WIN;
        }
        else if( _MovesPlayed == (_Rows * _Cols) )
        {
            _Result = RESULT_DRAW;

            for( int row = 0; row < _Rows; row++ )
                for( int col = 0; col < _Cols; col++ )
                    assert( PieceAt( row, col ) != 0 );
        }

        return true;
    }

    virtual int GetResult() const
    {
        return _Result;
    }

    virtual string Serialize() const
    {
        stringstream str;
        str << "(connect " << _RunLength << " ";

        for( int row = 0; row < _Rows; row++ )
        {
            if( row > 0 )
                str << '/';
            
            for( int col = 0; col < _Cols; col++ )
                str << ((PieceAt( row, col ) == 0)? '.' : ((PieceAt( row, col ) > 0)? 'R' : 'Y');
        }

        str << ')';
        return str;
    }

    virtual bool Deserialize( const string& ser )
    {
        string str = ser;
        boost::trim_all( str );

        if( str.length() < 2 )
            return false;

        str = str.substr( 1, str.length - 1 );

        vector< string > fields;
        boost::split( fields, str, ' ', boost::token_compress_on );

        if( fields.size() != 3 )
            return false;

        if( fields[0] != "connect" )
            return false;

        int runLength = std::stoi( fields[2].c_str() );
        if( runLength == 0 )
            return false;

        string grid = fields[2];
        vector< string > rowText;
        boost::split( rowText, fields[2], '/', boost::token_compress_on );

        if( rowText.length() < runLength )
            return false;

        int rows = rowText.length();
        if( rows < runLength )
            return false;
        if( rows < MAX_ROWS )
            return false;

        int cols = (int) rowText[0].length();
        if( cols > MAX_COLS )
            return false;

        for( int i = 1; i < rows; i++ )
            if( rowText[i].length() != cols )
                return false;

        this->Init( runLength, rows, cols );

        for( int row = 0; row < rows; row++ )
        {
            for( int col = 0; col < cols; col++ )
            {
                switch( rowText[row][col] )
                {
                    case 'R': PieceAt( row, col ) =  1; _MovesPlayed++; break;
                    case 'Y': PieceAt( row, col ) = -1; _MovesPlayed++; break;
                    case ' ': PieceAt( row, col ) =  0; break;
                    default:  return false;
                }
            }
        }
    }
}

private:
    bool InBounds( int row, int col )
    {
        if( (row >= 0) && (row < _Rows) && (col >= 0) && (col < _Cols) )
            return true;

        return false;
    }

    int8_t& PieceAt( int row, int col )
    {
        assert( InBounds( row, col ) );
        return _Slot[row * _Cols + col];
    }

    int CountRunLength( int row, int col, int dx, int dy )
    {
        assert( InBounds( row, col ) );
        if( !InBounds( row, col ) )
            return 0;

        int ourColor = _Slot[row][col];
        assert( ourColor != 0 );
        if( ourColor == 0 )
            return 0;

        int len = 1;
        for( int dir = 0; dir < 2; dir++ )
        {
            int x = row;
            int y = col;

            while( InBounds( x + dx, y + dy ) && (_Slot[x + dx][y + dy] == ourColor]) )
            {
                x += dx;
                y += dy;
                len++;
            }

            dx *= -1;
            dy *= -1;
        }

        assert( len <= _RunLength );

        return( len );
    }
};

