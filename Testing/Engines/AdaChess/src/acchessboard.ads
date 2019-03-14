--
--  AdaChess v2.0 - Simple Chess Engine
--
--  Copyright (C) 2013-2014 - Alessandro Iavicoli
--  Email: adachess@gmail.com - Web Page: http://www.adachess.com
--
--  This program is free software: you can redistribute it and/or modify
--  it under the terms of the GNU General Public License as published by
--  the Free Software Foundation, either version 3 of the License, or
--  (at your option) any later version.
--
--  This program is distributed in the hope that it will be useful,
--  but WITHOUT ANY WARRANTY; without even the implied warranty of
--  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--  GNU General Public License for more details.
--
--  You should have received a copy of the GNU General Public License
--  along with this program.  If not, see <http://www.gnu.org/licenses/>.



with Interfaces; use Interfaces;

package ACChessBoard is

   subtype Hash_Type is Unsigned_32;
   -- any position on the chessboard has to be hashed
   -- so we can fast perform a count on repetition due
   -- to the rules of 3-fold reps. This hash type is
   -- our 32-bit unsigned integer used to perform
   -- hashing with the Zobrist algorithm

   Hash       : Hash_Type;
   -- This is our current hash, updated any times a move is played
   -- and any times we load a new game from a FEN string

   type Castle_Type is
      record
	 White_Kingside  : Boolean;
	 White_Queenside : Boolean;
	 Black_Kingside  : Boolean;
	 Black_Queenside : Boolean;
      end record;
--     pragma Pack (Castle_Type);
   -- This castle flags helps to keep the status
   -- of castles while performing moving generation

   Castle     : Castle_Type := (False, False, False, False);
   No_Castle  : constant Castle_Type := (False, False, False, False);
   -- At any Ply, the Castle variable help us keeping the
   -- situation about castles.

   En_Passant : Integer := 0;
    -- The en-passant square at current Ply


   subtype Fifty_Counter_Type is Integer range 0 .. 101;
   Fifty      : Fifty_Counter_Type := 0;
   -- a 0 to 100 counter of half-moves due to respect the
   -- fifty moves rule in chess. When this counter exceed
   -- the value of 100 then the game ends with a Draw.


   subtype Piece_Type is Integer range -2 .. 13;
   -- Square-content descrition. In this range
   -- we have also frame (out-of-the-board), empty sqares
   -- and pieces


   type Piece_Position_Type is
      record
	 Square                 : Integer;
	 Piece                  : Piece_Type;
      end record;
--     pragma Pack (Piece_Position_Type);

   type Flag_Type is
      record
	 Capture              : Boolean := False;
	 Castle               : Boolean := False; -- moves rook after king
	 Pawn_Move            : Boolean := False; -- reset fifty-moves counter
	 Pawn_Two_Square_Move : Boolean := False; -- find en-passant square
	 En_Passant           : Boolean := False;
	 Pawn_Promote         : Boolean := False;
      end record;
--     pragma Pack (Flag_Type);
   -- flags are situation about a move. Like if a move is an en-passant
   -- or a pawn promotion

   No_Flags : constant Flag_Type := (False, False, False, False, False, False);

   procedure Reset (Flag : out Flag_Type) with Inline => True;
   procedure Reset_En_Passant with Inline => True;


   type Move_Type is
      record
	 From        : Integer := 0;
	 To          : Integer := 0;
	 Piece       : Piece_Type := 0;
	 Captured    : Piece_Type := 0; -- this is also the "non captured" standard piece
	 Score       : Integer := 0;
	 En_Passant  : Integer := 0;
	 Promotion   : Piece_Type := 0;
	 Castle      : Castle_Type;
	 Fifty       : Fifty_Counter_Type := 0;
	 Hash        : Hash_Type;
	 Flags       : Flag_Type := No_Flags;
      end record; --with Dynamic_Predicate => Move_Type.From /= Move_Type.To;
--     pragma Pack (Move_Type);
   -- Move description. I know: this is very heavy...

   function "=" (Left, Right : in Move_Type) return Boolean;
   -- Ada has built-in "/=" operator that now will use the ovveriding "=" operator

   No_Move : constant Move_Type := (0, 0, 0, 0, 0, 0, 0, No_Castle, 0, 0, No_Flags);

   type Ply_Type is array (1 .. 512) of Integer;
   -- Please note: 256 is up to a theoretical limit for pseudo-legal moves
   -- So as Ply_Type'Last use a value bigger than 256

   Moves_List    : array (Ply_Type'Range, Ply_Type'Range) of Move_Type; -- List of allowed moves, at any depth
   Moves_Counter : array (Ply_Type'Range) of Integer := (others =>  0); -- how many moves allowed here?
   History_Moves : array (1 .. 512) of Move_Type := (others => No_Move); -- Move played in the past
   Ply           : Integer := Ply_Type'First;
   History_Ply   : Integer := Ply_Type'First;
   History_Started_At : Integer := Ply_Type'First; -- used while loading FEN
   -- The difference between Ply and History_Ply is that Ply will be reset
   -- any time we call the "Think" function. Instead, History_Ply will never
   -- be reset (except on chessboard initialization)

   Engine : Integer;
   -- This is the color for the engine

   ---------------
   -- Constants --
   ---------------

   Frame	 : constant Integer := -2;
   Empty	 : constant Integer := -1;

   White	 : constant Integer := 0;
   Black	 : constant Integer := 1;

   White_Pawn	   : constant Integer := 2;
   Black_Pawn	   : constant Integer := 3;
   White_Knight	 : constant Integer := 4;
   Black_Knight	 : constant Integer := 5;
   White_Bishop	 : constant Integer := 6;
   Black_Bishop	 : constant Integer := 7;
   White_Rook	   : constant Integer := 8;
   Black_Rook	   : constant Integer := 9;
   White_Queen	  : constant Integer := 10;
   Black_Queen	  : constant Integer := 11;
   White_King	   : constant Integer := 12;
   Black_King    : constant Integer := 13;

   No_Piece  : constant Piece_Position_Type := (Empty, Empty);

   -- We want to keep an array only for piece positions.
   -- this will make us faster moving generation
   White_Pieces         : array (1 .. 16) of Piece_Position_Type;
   Black_Pieces         : array (1 .. 16) of Piece_Position_Type;
   White_Pieces_Counter : Integer := 16;
   Black_Pieces_Counter : Integer := 16;
   -- for each square, give the index of the array for white_pieces or black_pieces
   -- that hold the piece in that square (used in lookup piece)
   Piece_Table          : array (0 .. 143) of Integer := (others => Frame);


   -- Procedures and function that works with our piece positions

   procedure Add_White_Piece (Square, Piece : in Integer) with Inline => True;
   procedure Add_Black_Piece (Square, Piece : in Integer) with Inline => True;
   procedure Delete_White_Piece (Square : in Integer) with Inline => True;
   procedure Delete_Black_Piece (Square : in Integer) with Inline => True;
   procedure Update_White_Piece (From, To, Piece : in Integer) with Inline => True;
   procedure Update_Black_Piece (From, To, Piece : in Integer) with Inline => True;
   function Lookup_White_Piece (Square : in Integer) return Integer with Inline => True; -- dalla casa ritorna l'indice dell'array
   function Lookup_Black_Piece (Square : in Integer) return Integer with Inline => True; -- dalla casa ritorna l'indice dell'array;
   procedure Reset_Piece_Table;
   procedure Align_Piece_Table;
   procedure Display_Piece_Table;

   -- Our chessboard is described as an array of pieces
   -- To help finding move out of square we use a frame
   -- around the board.
   ChessBoard         : array (0 .. 143) of Integer := (others => Frame);

   subtype Side_To_Move_Type is Integer range White .. Black;

   History_Heuristic : array (Ply_Type'Range, ChessBoard'Range, ChessBoard'Range) of Integer;
   Killer_Heuristic_1  : array (Ply_Type'Range) of Move_Type;
   Killer_Heuristic_2  : array (Ply_Type'Range) of Move_Type;
   Killer_Heuristic_3  : array (Ply_Type'Range) of Move_Type;
   Killer_Score_1      : array (Ply_Type'Range) of Integer;
   Killer_Score_2      : array (Ply_Type'Range) of Integer;
   Killer_Score_3      : array (Ply_Type'Range) of Integer;
   -- Heuristic will help in move ordering

   procedure Update_Killer_Moves (Move : in Move_Type; Score : in Integer) with Inline => True;

   Side_To_Move : Side_To_Move_Type;
   Opponent     : Side_To_Move_Type;

   -- utilities functions
   Files : array (ChessBoard'Range) of Integer :=
     (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

   Ranks : array (ChessBoard'Range) of Integer :=
     (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,  8,  8,  8,  8,  8,  8,  8,  8, -1, -1,
      -1, -1,  7,  7,  7,  7,  7,  7,  7,  7, -1, -1,
      -1, -1,  6,  6,  6,  6,  6,  6,  6,  6, -1, -1,
      -1, -1,  5,  5,  5,  5,  5,  5,  5,  5, -1, -1,
      -1, -1,  4,  4,  4,  4,  4,  4,  4,  4, -1, -1,
      -1, -1,  3,  3,  3,  3,  3,  3,  3,  3, -1, -1,
      -1, -1,  2,  2,  2,  2,  2,  2,  2,  2, -1, -1,
      -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

   Diagonals : array (ChessBoard'Range) of Integer :=
     (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,  7,  6,  5,  4,  3,  2,  1,  0, -1, -1,
      -1, -1,  6,  5,  4,  3,  2,  1,  0,  8, -1, -1,
      -1, -1,  5,  4,  3,  2,  1,  0,  8,  9, -1, -1,
      -1, -1,  4,  3,  2,  1,  0,  8,  9, 10, -1, -1,
      -1, -1,  3,  2,  1,  0,  8,  9, 10, 11, -1, -1,
      -1, -1,  2,  1,  0,  8,  9, 10, 11, 12, -1, -1,
      -1, -1,  1,  0,  8,  9, 10, 11, 12, 13, -1, -1,
      -1, -1,  0,  8,  9, 10, 11, 12, 13, 14, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

   Anti_Diagonals : array (ChessBoard'Range) of Integer :=
     (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1,  7,  8,  9, 10, 11, 12, 13, 14, -1, -1,
      -1, -1,  6,  7,  8,  9, 10, 11, 12, 13, -1, -1,
      -1, -1,  5,  6,  7,  8,  9, 10, 11, 12, -1, -1,
      -1, -1,  4,  5,  6,  7,  8,  9, 10, 11, -1, -1,
      -1, -1,  3,  4,  5,  6,  7,  8,  9, 10, -1, -1,
      -1, -1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1,
      -1, -1,  1,  2,  3,  4,  5,  6,  7,  8, -1, -1,
      -1, -1,  0,  1,  2,  3,  4,  5,  6,  7, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

   function File (Square : in Integer) return Integer with Inline => True;
   function Rank (Square : in Integer) return Integer with Inline => True;

   function Diagonal (Square : in Integer) return Integer with Inline => True;
   function Anti_Diagonal (Square : in Integer) return Integer with Inline => True;

   Pinning_Piece_Table : array (Piece_Type'Range, Piece_Type'Range) of Boolean;
--     Forking_Piece_Table : array (Piece_Type'Range, Piece_Type'Range) of Boolean;

--     White_Attack_Board : array (0 .. 143) of Integer;
--     White_Defend_Board : array (0 .. 143) of Integer;
--     Black_Attack_Board : array (0 .. 143) of Integer;
--     Black_Defend_Board : array (0 .. 143) of Integer;


   -- other utilities functions
   function Is_White (Piece : in Integer) return Boolean with Inline => True;
   function Is_Black (Piece : in Integer) return Boolean with Inline => True;
   function Is_In_Board (Square : in Integer) return Boolean with Inline => True;
   function Is_Piece (Piece : in Integer) return Boolean with Inline => True;
   function Is_Empty (Square  : in Integer) return Boolean with Inline => True;
   function Is_Capture (Move : in Move_Type) return Boolean with Inline => True;

   function Last_Move_Played return Move_Type with Inline => True;

   -- The Test_Validity helps to looks for king in check only
   -- when needed. We will looks for moves that CAN leave king
   -- in check and we don't looks for moves that CAN'T leave
   -- the king in check. To do this, we looks only on moves that are
   -- 1) Castling
   -- 2) En-passant
   -- 3) Moves with absolute pinned piece
   -- in the case of 3) we use a semplified algorithm that looks
   -- on every move from same rank/file/diagonal/anti_diagonal
   -- to another rank/file/diagonal/anti_diagonal
   -- The Force_Test_Validity force the test to be exectued on every
   -- move generator (to be used when king is in check!)
   Force_Test_Validity : Boolean;
   procedure Clear_Moves_List with Inline => True;
   procedure Register_Move (Move : in Move_Type; Test_Validity : in Boolean) with Inline => True;
   procedure Register_Move (From, To : in Integer; Test_Validity : Boolean) with Inline => True;
   procedure Register_Move (From, To : in Integer; Flags : in Flag_Type; Test_Validity : Boolean) with Inline => True;

   MVV_LVA_Table : array (Piece_Type'Range, Piece_Type'Range) of Integer;

   procedure MVV_LVA (Move : in out Move_Type) with Inline => True;
   procedure Assign_Score (Move : in out Move_Type) renames MVV_LVA;

   -----------------------------------------------
   -- Offsets for moving piece around the board --
   -----------------------------------------------

   North            : constant Integer := -12;
   North_North_East : constant Integer := -23;
   North_East       : constant Integer := -11;
   North_East_East  : constant Integer := -10;
   East             : constant Integer := +1;
   South_East_East  : constant Integer := +14;
   South_East       : constant Integer := +13;
   South_South_East : constant Integer := +25;
   South            : constant Integer := +12;
   South_South_West : constant Integer := +23;
   South_West       : constant Integer := +11;
   South_West_West  : constant Integer := +10;
   West             : constant Integer := -1;
   North_West_West  : constant Integer := -14;
   North_West       : constant Integer := -13;
   North_North_West : constant Integer := -25;

   Knight_Offsets : array (1 .. 8) of Integer :=
     (North_North_East, North_East_East,
      South_East_East, South_South_East,
      South_South_West, South_West_West,
      North_West_West, North_North_West);

   Bishop_Offsets : array (1 .. 4) of Integer :=
     (North_West, North_East, South_East, South_West);

   Rook_Offsets   : array (1 .. 4) of Integer :=
     (North, East, South, West);

   Queen_Offsets  : array (1 .. 8) of Integer :=
     (North, North_East, East, South_East,
      South, South_West, West, North_West);

   King_Offsets   : array (1 .. 8) of Integer :=
     (North, North_East, East, South_East,
      South, South_West, West, North_West);

   ----------------------
   -- Square constants --
   ----------------------

   A8 : constant Integer := ChessBoard'First + 26;
   B8 : constant Integer := ChessBoard'First + 27;
   C8 : constant Integer := ChessBoard'First + 28;
   D8 : constant Integer := ChessBoard'First + 29;
   E8 : constant Integer := ChessBoard'First + 30;
   F8 : constant Integer := ChessBoard'First + 31;
   G8 : constant Integer := ChessBoard'First + 32;
   H8 : constant Integer := ChessBoard'First + 33;

   A7 : constant Integer := ChessBoard'First + 38;
   B7 : constant Integer := ChessBoard'First + 39;
   C7 : constant Integer := ChessBoard'First + 40;
   D7 : constant Integer := ChessBoard'First + 41;
   E7 : constant Integer := ChessBoard'First + 42;
   F7 : constant Integer := ChessBoard'First + 43;
   G7 : constant Integer := ChessBoard'First + 44;
   H7 : constant Integer := ChessBoard'First + 45;

   A6 : constant Integer := ChessBoard'First + 50;
   B6 : constant Integer := ChessBoard'First + 51;
   C6 : constant Integer := ChessBoard'First + 52;
   D6 : constant Integer := ChessBoard'First + 53;
   E6 : constant Integer := ChessBoard'First + 54;
   F6 : constant Integer := ChessBoard'First + 55;
   G6 : constant Integer := ChessBoard'First + 56;
   H6 : constant Integer := ChessBoard'First + 57;

   A5 : constant Integer := ChessBoard'First + 62;
   B5 : constant Integer := ChessBoard'First + 63;
   C5 : constant Integer := ChessBoard'First + 64;
   D5 : constant Integer := ChessBoard'First + 65;
   E5 : constant Integer := ChessBoard'First + 66;
   F5 : constant Integer := ChessBoard'First + 67;
   G5 : constant Integer := ChessBoard'First + 68;
   H5 : constant Integer := ChessBoard'First + 69;

   A4 : constant Integer := ChessBoard'First + 74;
   B4 : constant Integer := ChessBoard'First + 75;
   C4 : constant Integer := ChessBoard'First + 76;
   D4 : constant Integer := ChessBoard'First + 77;
   E4 : constant Integer := ChessBoard'First + 78;
   F4 : constant Integer := ChessBoard'First + 79;
   G4 : constant Integer := ChessBoard'First + 80;
   H4 : constant Integer := ChessBoard'First + 81;

   A3 : constant Integer := ChessBoard'First + 86;
   B3 : constant Integer := ChessBoard'First + 87;
   C3 : constant Integer := ChessBoard'First + 88;
   D3 : constant Integer := ChessBoard'First + 89;
   E3 : constant Integer := ChessBoard'First + 90;
   F3 : constant Integer := ChessBoard'First + 91;
   G3 : constant Integer := ChessBoard'First + 92;
   H3 : constant Integer := ChessBoard'First + 93;

   A2 : constant Integer := ChessBoard'First + 98;
   B2 : constant Integer := ChessBoard'First + 99;
   C2 : constant Integer := ChessBoard'First + 100;
   D2 : constant Integer := ChessBoard'First + 101;
   E2 : constant Integer := ChessBoard'First + 102;
   F2 : constant Integer := ChessBoard'First + 103;
   G2 : constant Integer := ChessBoard'First + 104;
   H2 : constant Integer := ChessBoard'First + 105;

   A1 : constant Integer := ChessBoard'First + 110;
   B1 : constant Integer := ChessBoard'First + 111;
   C1 : constant Integer := ChessBoard'First + 112;
   D1 : constant Integer := ChessBoard'First + 113;
   E1 : constant Integer := ChessBoard'First + 114;
   F1 : constant Integer := ChessBoard'First + 115;
   G1 : constant Integer := ChessBoard'First + 116;
   H1 : constant Integer := ChessBoard'First + 117;

   White_King_Position : Integer := E1;
   Black_King_Position : Integer := E8;


   procedure Initialize; -- init the baord!
   procedure Display;
   procedure Change_Side_To_Move with Inline => True;


   procedure Generate_Moves;
   procedure Generate_White_Pawn_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Knight_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Bishop_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Rook_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Queen_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_King_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Pawn_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Knight_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Bishop_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Rook_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Queen_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_King_Moves (Index, Square : in Integer) with Inline => True;

   procedure Generate_Capture_Moves;
   procedure Generate_White_Pawn_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Knight_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Bishop_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Rook_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_Queen_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_White_King_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Pawn_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Knight_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Bishop_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Rook_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_Queen_Capture_Moves (Index, Square : in Integer) with Inline => True;
   procedure Generate_Black_King_Capture_Moves (Index, Square : in Integer) with Inline => True;


   function Has_King_In_Check (Side : in Integer) return Boolean with Inline => True;
   -- this is a fast version by looking for attackers using king square as starting square

   function Find_Attack (Target_Square, Side : in Integer) return Boolean;
   -- Does some piece of side Side attacks Target_Square?

   function Parse_Input_Move (Input : in String) return Move_Type;

   procedure Echo (M : in Move_Type);
   function Echo2 (M : in Move_Type) return String;

   function Play (Move : in Move_Type) return Boolean;
   procedure Undo;
   procedure Undo (Times : in Natural);
   function Is_Valid (Move : in Move_Type) return Boolean;

   Pc_Sqr    : constant array (0 .. 143) of String (1 .. 2) :=
     ("  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ",
      "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ",
      "  ", "  ", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8", "  ", "  ",
      "  ", "  ", "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "  ", "  ",
      "  ", "  ", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6", "  ", "  ",
      "  ", "  ", "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "  ", "  ",
      "  ", "  ", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4", "  ", "  ",
      "  ", "  ", "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "  ", "  ",
      "  ", "  ", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", "  ", "  ",
      "  ", "  ", "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "  ", "  ",
      "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ",
      "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ");


   ------------------------------------
   -- Debug functions and procedures --
   ------------------------------------

   procedure Put_White_Pieces;
   procedure Put_Black_Pieces;
   procedure Put_Moves_List;

end ACChessBoard;
