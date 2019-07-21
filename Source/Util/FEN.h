// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

static void PositionToString( const Position& pos, char* str )
{
    if( pos._BoardFlipped )
    {
        Position flipped;
        flipped.FlipFrom( pos );

        PositionToString( flipped, str );
        return;
    }

    char* cur = str;

    for( u64 bit = 1ULL << 63; bit != 0; bit >>= 1 )
    {
        if( pos._WhitePawns   & bit )   *cur++ = 'P'; else
        if( pos._WhiteKnights & bit )   *cur++ = 'N'; else
        if( pos._WhiteBishops & bit )   *cur++ = 'B'; else
        if( pos._WhiteRooks   & bit )   *cur++ = 'R'; else
        if( pos._WhiteQueens  & bit )   *cur++ = 'Q'; else
        if( pos._WhiteKing    & bit )   *cur++ = 'K'; else
        if( pos._BlackPawns   & bit )   *cur++ = 'p'; else
        if( pos._BlackKnights & bit )   *cur++ = 'n'; else
        if( pos._BlackBishops & bit )   *cur++ = 'b'; else
        if( pos._BlackRooks   & bit )   *cur++ = 'r'; else
        if( pos._BlackQueens  & bit )   *cur++ = 'q'; else
        if( pos._BlackKing    & bit )   *cur++ = 'k'; else
        {
            if( (cur > str) && (cur[-1] >= '1') && (cur[-1] < '8') )
                cur[-1]++;
            else
                *cur++ = '1';
        }

        if( bit & FILE_H )
            *cur++ = (bit & RANK_1)? ' ' : '/';
    }

    str = cur;

    *str++ = pos._WhiteToMove? 'w' : 'b';
    *str++ = ' ';

    if(   pos._CastlingAndEP & SQUARE_H1 )     *str++ = 'K';
    if(   pos._CastlingAndEP & SQUARE_A1 )     *str++ = 'Q';
    if(   pos._CastlingAndEP & SQUARE_H8 )     *str++ = 'k';
    if(   pos._CastlingAndEP & SQUARE_A8 )     *str++ = 'q';
    if( !(pos._CastlingAndEP & CASTLE_ROOKS) ) *str++ = '-';

    *str++ = ' ';

    if( pos._CastlingAndEP & EP_SQUARES )
    {
        int idx = (int) LowestBitIndex( (u64) (pos._CastlingAndEP & EP_SQUARES) );
        *str++ = "hgfedcba"[idx & 7];
        *str++ = "12345678"[idx / 8]; 
        *str++ = ' ';
    }
    else
    {
        *str++ = '-';
        *str++ = ' ';
    }

    sprintf( str, "%d %d", (int) pos._HalfmoveClock, (int) pos._FullmoveNum );
}


static const char* StringToPosition( const char* str, Position& pos )
{
    memset( &pos, 0, sizeof( pos ) );

    for( u64 bit = 1ULL << 63; bit != 0; bit >>= 1 )
    {
        switch( *str++ )
        {
        case 'P':   pos._WhitePawns   |= bit; break;
        case 'N':   pos._WhiteKnights |= bit; break;
        case 'B':   pos._WhiteBishops |= bit; break;
        case 'R':   pos._WhiteRooks   |= bit; break;
        case 'Q':   pos._WhiteQueens  |= bit; break;
        case 'K':   pos._WhiteKing    |= bit; break;
        case 'p':   pos._BlackPawns   |= bit; break;
        case 'n':   pos._BlackKnights |= bit; break;
        case 'b':   pos._BlackBishops |= bit; break;
        case 'r':   pos._BlackRooks   |= bit; break;
        case 'q':   pos._BlackQueens  |= bit; break;
        case 'k':   pos._BlackKing    |= bit; break;
        case '8':   bit >>= 7;  break;
        case '7':   bit >>= 6;  break;
        case '6':   bit >>= 5;  break;
        case '5':   bit >>= 4;  break;
        case '4':   bit >>= 3;  break;
        case '3':   bit >>= 2;  break;
        case '2':   bit >>= 1;  break;
        case '1':               break;
        case '/':   bit <<= 1;  break;
        default:    return( NULL );
        }
    }

    if( *str == ' ' ) str++;

    switch( *str++ )
    {
    case 'w':   pos._WhiteToMove = 1; break;
    case 'b':   pos._WhiteToMove = 0; break;
    default:    return( NULL );
    }

    if( *str == ' ' ) 
        str++;

    while( *str && (*str != ' ') )
    {
        if( (*str == 'K') || (*str == 'H')) pos._CastlingAndEP |= SQUARE_H1; else
        if( (*str == 'Q') || (*str == 'A')) pos._CastlingAndEP |= SQUARE_A1; else
        if( (*str == 'k') || (*str == 'h')) pos._CastlingAndEP |= SQUARE_H8; else
        if( (*str == 'q') || (*str == 'a')) pos._CastlingAndEP |= SQUARE_A8; else
        if( *str != '-' )                   return( NULL );

        str++;
    }

    str++;

    if( *str == '-' )
        str++;
    else
    {
        int file  = 'h' - tolower( *str++ );
        int rank  = *str++ - '1';

        if( (file | rank) & ~7 )
            return( NULL );

        int idx = (rank * 8) + file;
        pos._CastlingAndEP |= SquareBit( idx );
    }

    if( *str == '-' )
        str++;
    else
    {
        pos._HalfmoveClock = 0;
        while( isdigit( *str ) )
            pos._HalfmoveClock = (pos._HalfmoveClock * 10) + (*str++ - '0');

        if( *str == '-' )
            str++;
        else
        {
            pos._FullmoveNum = 0;
            while( isdigit( *str ) )
                pos._FullmoveNum = (pos._FullmoveNum * 10) + (*str++ - '0');
        }
    }

    if( !pos._WhiteToMove )
        pos.FlipInPlace();

    while( *str && (*str == ' ') )
        str++;
            
    return( str );
}



static void MoveSpecToString( const MoveSpec& spec, char* str ) 
{
    *str++ = "hgfedcba"[spec._Src & 7];
    *str++ = "12345678"[spec._Src / 8];
    *str++ = "hgfedcba"[spec._Dest & 7];
    *str++ = "12345678"[spec._Dest / 8];
    *str++ = spec.GetPromoteChar();
    *str++ = 0;
}


static const char* StringToMoveSpec( const char* str, MoveSpec& spec )
{
    int src_file0  = 'h' - tolower( *str++ );
    int src_rank0  = *str++ - '1';
    int dest_file0 = 'h' - tolower( *str++ );
    int dest_rank0 = *str++ - '1';

    if( (src_file0 | src_rank0 | dest_file0 | dest_rank0) & ~7 )
        return( NULL );

    spec._Src  = (src_rank0  * 8) + src_file0;
    spec._Dest = (dest_rank0 * 8) + dest_file0;
    spec._Type = 0;

    switch( tolower( *str ) )
    {
    case 'n':   spec._Type = PROMOTE_KNIGHT; str++; break;
    case 'b':   spec._Type = PROMOTE_BISHOP; str++; break;
    case 'r':   spec._Type = PROMOTE_ROOK;   str++; break;
    case 'q':   spec._Type = PROMOTE_QUEEN;  str++; break;
    }

    while( *str && strchr( "-+#!?", *str ) ) 
        str++;

    if( *str && !isspace( *str++ ) )    
        return( NULL );

    return( str );
}


static string SerializePosition( const Position& pos )
{
    char fen[MAX_FEN_LENGTH];
    PositionToString( pos, fen );

    return string( fen );
}

static Position DeserializePosition( const string& str )
{
    Position pos;
    StringToPosition( str.c_str(), pos );

    return pos;
}

static string SerializeMoveSpec( const MoveSpec& spec )
{
    char movetext[MAX_MOVETEXT_LENGTH];
    MoveSpecToString( spec, movetext );

    return string( movetext );
}

static string SerializeMoveList( const MoveList& moves )
{
    char result[(MAX_MOVETEXT_LENGTH + 1) * MAX_POSSIBLE_MOVES];
    char* cursor = result;

    *cursor = 0;
    for( int i = 0; i < moves._Count; i++ )
    {
        if( i > 0 )
            *cursor++ = ' ';

        MoveSpecToString( moves._Move[i], cursor );
        while( *cursor )
            cursor++;
    }

    return string( result );
}

