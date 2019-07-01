// JAGLAVAK CHESS ENGINE (c) 2019 Stuart Riffle
#pragma once

enum
{
    H1, G1, F1, E1, D1, C1, B1, A1,
    H2, G2, F2, E2, D2, C2, B2, A2,
    H3, G3, F3, E3, D3, C3, B3, A3,
    H4, G4, F4, E4, D4, C4, B4, A4,
    H5, G5, F5, E5, D5, C5, B5, A5,
    H6, G6, F6, E6, D6, C6, B6, A6,
    H7, G7, F7, E7, D7, C7, B7, A7,
    H8, G8, F8, E8, D8, C8, B8, A8,
};

enum
{
    WHITE,
    BLACK
};

enum
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,

    NUM_PIECE_TYPES,
};

enum 
{
    PROMOTE_NONE,

    PROMOTE_KNIGHT,
    PROMOTE_BISHOP,
    PROMOTE_ROOK,
    PROMOTE_QUEEN
};

enum
{
    SHIFT_N    =  8,
    SHIFT_NW   =  9,
    SHIFT_W    =  1,
    SHIFT_SW   = -7,
    SHIFT_S    = -8,
    SHIFT_SE   = -9,
    SHIFT_E    = -1,
    SHIFT_NE   =  7,
};

const PDECL int     MAX_POSSIBLE_MOVES      = 218;
const PDECL int     MAX_FEN_LENGTH          = 96;
const PDECL int     MAX_MOVETEXT_LENGTH     = 6;

const PDECL u64     SQUARE_A1               = 1ULL << A1;
const PDECL u64     SQUARE_A8               = 1ULL << A8;
const PDECL u64     SQUARE_B1               = 1ULL << B1;
const PDECL u64     SQUARE_B8               = 1ULL << B8;
const PDECL u64     SQUARE_C1               = 1ULL << C1;
const PDECL u64     SQUARE_C8               = 1ULL << C8;
const PDECL u64     SQUARE_D1               = 1ULL << D1;
const PDECL u64     SQUARE_D8               = 1ULL << D8;
const PDECL u64     SQUARE_E1               = 1ULL << E1;
const PDECL u64     SQUARE_E8               = 1ULL << E8;
const PDECL u64     SQUARE_F1               = 1ULL << F1;
const PDECL u64     SQUARE_F8               = 1ULL << F8;
const PDECL u64     SQUARE_G1               = 1ULL << G1;
const PDECL u64     SQUARE_G8               = 1ULL << G8;
const PDECL u64     SQUARE_H1               = 1ULL << H1;
const PDECL u64     SQUARE_H8               = 1ULL << H8;

const PDECL u64     FILE_A                  = 0x8080808080808080ULL;
const PDECL u64     FILE_B                  = 0x4040404040404040ULL;
const PDECL u64     FILE_C                  = 0x2020202020202020ULL;
const PDECL u64     FILE_D                  = 0x1010101010101010ULL;
const PDECL u64     FILE_E                  = 0x0808080808080808ULL;
const PDECL u64     FILE_F                  = 0x0404040404040404ULL;
const PDECL u64     FILE_G                  = 0x0202020202020202ULL;
const PDECL u64     FILE_H                  = 0x0101010101010101ULL;

const PDECL u64     RANK_1                  = 0x00000000000000FFULL;
const PDECL u64     RANK_2                  = 0x000000000000FF00ULL;
const PDECL u64     RANK_3                  = 0x0000000000FF0000ULL;
const PDECL u64     RANK_4                  = 0x00000000FF000000ULL;
const PDECL u64     RANK_5                  = 0x000000FF00000000ULL;
const PDECL u64     RANK_6                  = 0x0000FF0000000000ULL;
const PDECL u64     RANK_7                  = 0x00FF000000000000ULL;
const PDECL u64     RANK_8                  = 0xFF00000000000000ULL;

const PDECL u64     LIGHT_SQUARES           = 0x55AA55AA55AA55AAULL;
const PDECL u64     DARK_SQUARES            = 0xAA55AA55AA55AA55ULL;
const PDECL u64     ALL_SQUARES             = 0xFFFFFFFFFFFFFFFFULL;

const PDECL u64     CASTLE_ROOKS            = SQUARE_A1 | SQUARE_H1 | SQUARE_A8 | SQUARE_H8;
const PDECL u64     EP_SQUARES              = RANK_3 | RANK_6;
const PDECL u64     CORNER_SQUARES          = (FILE_A | FILE_H) & (RANK_1 | RANK_8);

template< typename T > struct   MoveSpecT;
typedef MoveSpecT< u8 >         MoveSpec;

template< typename T > struct   MoveMapT;
typedef MoveMapT< u64 >         MoveMap;

template< typename T > struct   PositionT;
typedef PositionT< u64 >        Position;

