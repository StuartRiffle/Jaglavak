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



with ACChessBoard; use ACChessBoard;

package ACEvaluate is

   type Game_Status_Type is (Opening, Middle_Game, End_Game);
   Game_Status : Game_Status_Type;

   Pawn_Score      : constant Integer := 100;
   Knight_Score    : constant Integer := 310;
   Bishop_Score    : constant Integer := 325;
   Rook_Score      : constant Integer := 550;
   Queen_Score     : constant Integer := 980;
   King_Score      : constant Integer := 1_000_000;

   CheckMate    : constant Integer := 990_990; -- should be less than king score, but very close to
   Draw         : constant Integer := 0;

   Count_White_Pieces : Integer;
   Count_Black_Pieces : Integer;
   Material_Advantage_Recapture_Encourage : constant Integer := 10;

   Endgame_Score_Limit : constant Integer := 4200;

   Isolated_Pawn_Penalty          : constant Integer := 10;
   Doubled_Pawn_Penalty           : constant Integer := 5;
   Blocked_Pawn_Penalty           : constant Integer := 5;
   Partially_Blocked_Pawn_Penalty : constant Integer := 2;
   Backward_Pawn_Penalty          : constant Integer := 5; -- ok
   Passed_Pawn_Bonus              : constant Integer := 10;
   Passed_Pawn_Score              : constant Integer := 20;
   Pawn_Blocked_On_Center_Penalty : constant Integer := 15;
   Pawn_Free_To_Kill_Penalty      : constant Integer := 30; -- not disturbed opponent center pawn
   Unprotected_Pawn_On_End_Game   : constant Integer := 15;

   -- give a small bonus for the bishop pairs
   Bishop_Pairs_Bonus             : constant Integer := 10;
   Trapped_Bishop_Penalty         : constant Integer := 16;
   Trapped_Bishop_Half_Penalty    : constant Integer := Trapped_Bishop_Penalty / 2;
   Blocked_Bishop_Penalty         : constant Integer := 10;

   Blocked_Rook_Penalty        : constant Integer := 20;
   Rook_On_Open_File_Bonus        : constant Integer := 20;
   Rook_On_Semi_Open_File_Bonus   : constant Integer := 10;
   Rook_On_Seventh_Rank_Bonus     : constant Integer := 35;
   Rooks_On_Same_File_Bonus        : constant Integer := 10;
   Rooks_On_Same_Rank_Bonus        : constant Integer := 10;

   Fork_Bonus                     : Integer := 25;
   Hanged_Piece_Penalty           : Integer := 10;

   Occupying_Weak_Square_Bonus    : constant Integer := 25;

   Protected_Piece_Bonus : constant Integer := 5;
   Unprotected_Piece_Penalty      : constant Integer := 8;
   Dangerous_Unprotected_Piece_Penalty : constant Integer := 15;
   Piece_Mobility_Bonus                : constant Integer := 2;

   -- Score for pinning pieces
   Pinning_Piece_Bonus                  : constant Integer := 15;
   Unprotected_Pinned_Piece_Penalty     : constant Integer := 25;
   Discovery_Check_Threat_Bonus         : constant Integer := 30;

   --     King_Not_Castled_Penalty : Integer := 45;
   King_Has_Castled_Bonus                 : constant Integer := 30;
   King_Has_Moved_Before_Castling_Penalty : constant Integer := 10;
   King_Castled_Bonus                     : constant Integer := 0;
   King_Castle_Safety                     : constant Integer := 10;
   King_Has_Castled_Protection_Bonus      : constant Integer := 15;
   King_Has_Castled_With_Fianchetto_Bonus : constant Integer := 10;
   King_Has_Castled_With_Hole_Bonus       : constant Integer := 5;
   King_Has_Castled_Half_Bonus            : constant Integer := 7;
   King_Has_Castled_Corrupted_Bonus       : constant Integer := 4;
   King_Castle_Expose_Penalty             : constant Integer := 2;
   King_Castle_Without_A_Pawn_Penalty     : constant Integer := 35;

   King_Protected_By_Knight_Bonus         : constant Integer := 10;
   King_Protected_By_Friendly_Bonus          : constant Integer := 5;

   Not_Castled                  : constant Integer := 0;
   White_King_Castled_Kingside  : constant Integer := 1;
   White_King_Castled_Queenside : constant Integer := 2;
   Black_King_Castled_Kingside  : constant Integer := 3;
   Black_King_Castled_Queenside : constant Integer := 4;


   White_Pawn_Position : array (1 .. 8) of Integer;
   Black_Pawn_Position : array (1 .. 8) of Integer;
   White_Pawn_Counter  : Integer;
   Black_Pawn_Counter  : Integer;
   White_Pawn_Rank     : array (0 .. 9) of Integer := (others => 0);
   Black_Pawn_Rank     : array (0 .. 9) of Integer := (others => 0);
   White_Pawn_File     : array (1 .. 8) of Boolean;
   Black_Pawn_File     : array (1 .. 8) of Boolean;

   White_Rooks_Position : array (1 .. 10) of Integer;
   Black_Rooks_Position : array (1 .. 10) of Integer;
   White_Rooks_Counter  : Integer;
   Black_Rooks_Counter  : Integer;

   White_Knights_Position : array (1 .. 10) of Integer;
   Black_Knights_Position : array (1 .. 10) of Integer;
   White_Knights_Counter : Integer;
   Black_Knights_Counter : Integer;

   White_Bishops_Position : array (1 .. 10) of Integer;
   Black_Bishops_Position : array (1 .. 10) of Integer;
   White_Bishops_Counter : Integer;
   Black_Bishops_Counter  : Integer;

   White_Bishops_Color    : array (White .. Black) of Boolean;
   Black_Bishops_Color    : array (White .. Black) of Boolean;

   -- assuming no queen or 1 queen
   -- when there are 2 or more queens
   -- it become not important to trace
   -- where they are. Just one is needed
   White_Queen_Position   : Integer;
   Black_Queen_Position   : Integer;

   White_Weak_Board         : array (ChessBoard'Range) of Boolean;
   Black_Weak_Board         : array (ChessBoard'Range) of Boolean;

   Weak_Square_Potential_Damage_Penalty         : constant Integer := 25;
   Friendly_Piece_On_Opponent_Weak_Square_Bonus : constant Integer := 30;

   Queen_Moves_On_First_Moves_Penalty           : constant Integer := 5;


   Pawn_Square_Value : constant array (ChessBoard'Range) of Integer
     := (
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 5, 10, 15, 20, 20, 15, 10, 5, 0, 0,
	 0, 0, 4, 8, 12, 16, 16, 12, 8, 4, 0, 0,
	 0, 0, 3, 6, 9, 14, 14, 9, 6, 3, 0, 0,
	 0, 0, 2, 4, 6, 12, 12, 6, 4, 2, 0, 0,
	 0, 0, 1, 2, 3, 10, 10, 3, 2, 1, 0, 0,
	 0, 0, 0, 0, 0, -30,-30, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);


   Knight_Square_Value : constant array (ChessBoard'Range) of Integer
     := (
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, -20, -10, -10, -10, -10, -10, -10, -20, 0, 0,
	 0, 0, -10,   0,   0,   5,   5,   0,   0, -10, 0, 0,
	 0, 0, -10,   0,  15,  15,  15,  15,   0, -10, 0, 0,
	 0, 0, -10,   0,  15,  20,  20,  15,   0, -10, 0, 0,
	 0, 0, -10,   0,  15,  20,  20,  15,   0, -10, 0, 0,
	 0, 0, -10,   0,  15,  15,  15,  15,   0, -10, 0, 0,
	 0, 0, -10,   0,   0,   5,   5,   0,   0, -10, 0, 0,
	 0, 0, -20, -10, -10, -10, -10, -10, -10, -20, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);


   Bishop_Square_Value : constant array (ChessBoard'Range) of Integer
     := (
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, -2, -2, -2, -2, -2, -2, -2, -2, 0, 0,
	 0, 0, -2, 8, 5, 5, 5, 5, 8, -2, 0, 0,
	 0, 0, -2, 3, 3, 5, 5, 3, 3, -2, 0, 0,
	 0, 0, -2, 2, 5, 4, 4, 5, 2, -2, 0, 0,
	 0, 0, -2, 2, 5, 4, 4, 5, 2, -2, 0, 0,
	 0, 0, -2, 3, 3, 5, 5, 3, 3, -2, 0, 0,
	 0, 0, -2, 8, 5, 5, 5, 5, 8, -2, 0, 0,
	 0, 0, -2, -2, -2, -2, -2, -2, -2, -2, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

--     Rook_Square_Value   : constant array (ChessBoard'Range) of Integer
--       := (
--  	 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0,
--  	 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0,
--  	 0, 0, 10, 15, 20, 25, 25, 20, 15, 10, 0, 0,
--  	 0, 0,  0, 10, 15, 20, 20, 15, 10,  0, 0, 0,
--  	 0, 0, -20, -20, -20, -20, -20, -20, -20, -20, 0, 0,
--  	 0, 0, -20, -20, -20, -30, -30, -20, -20, -20, 0, 0,
--  	 0, 0, -20, -20, -20, -20, -20, -20, -20, -20, 0, 0,
--  	 0, 0, -15, -15, -15, -10, -10, -15, -15, -15, 0, 0,
--  	 0, 0,  0,  0,  0,  7, 10,  0,  0,  0, 0, 0,
--  	 0, 0,  2,  2,  2,  2,  2,  2,  2,  2, 0, 0,
--  	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
--  	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

   Queen_Square_Value : constant array (ChessBoard'Range) of Integer
     := (
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 5, 5, 5, 10, 10, 5, 5, 5, 0, 0,
	 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0,
	 0, 0, -30, -30, -30, -30, -30, -30, -30, -30, 0, 0,
	 0, 0, -60, -40, -40, -60, -60, -40, -40, -60, 0, 0,
	 0, 0, -40, -40, -40, -40, -40, -40, -40, -40, 0, 0,
	 0, 0, -15, -15, -15, -10, -10, -15, -15, -15, 0, 0,
	 0, 0, 0, 0, 0, 7, 10, 5, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);



   King_Square_Value   : constant array (ChessBoard'Range) of Integer
     := (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, -55, -55, -89, -89, -89, -89, -55, -55, 0, 0,
	 0, 0, -34, -34, -55, -55, -55, -55, -34, -34, 0, 0,
	 0, 0, -21, -21, -34, -34, -34, -34, -21, -21, 0, 0,
	 0, 0, -13, -13, -21, -21, -21, -21, -13, -13, 0, 0,
	 0, 0, -8, -8, -13, -13, -13, -13, -8, -8, 0, 0,
	 0, 0, -5, -5, -8, -8, -8, -8, -5, -5, 0, 0,
	 0, 0, -3, -5, -6, -6, -6, -6, -5, -3, 0, 0,
	 0, 0, 2, 14, 0, 0, 0, 9, 14, 2, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

   King_End_Game_Square_Value : constant array (ChessBoard'Range) of Integer
     := (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, -50, -30, -20, -20, -20, -20, -30, -50, 0, 0,
	 0, 0, -30,  0, 10, 10, 10, 10,  0, -30, 0, 0,
	 0, 0, -20, 10, 25, 25, 25, 25, 10, -20, 0, 0,
	 0, 0, -20, 10, 25, 50, 50, 25, 10, -20, 0, 0,
	 0, 0, -20, 10, 25, 50, 50, 25, 10, -20, 0, 0,
	 0, 0, -20, 10, 25, 25, 25, 25, 10, -20, 0, 0,
	 0, 0, -30,  0, 10, 10, 10, 10,  0, -30, 0, 0,
	 0, 0, -50, -30, -20, -20, -20, -20, -30, -50, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);



   -- flip board to match <piece>SquareValue for Black side
   Flip : constant array (ChessBoard'Range) of Integer
     := (0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0,
	 0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0,
	 0, 0, A1, B1, C1, D1, E1, F1, G1, H1, 0, 0,
	 0, 0, A2, B2, C2, D2, E2, F2, G2, H2, 0, 0,
	 0, 0, A3, B3, C3, D3, E3, F3, G3, H3, 0, 0,
	 0, 0, A4, B4, C4, D4, E4, F4, G4, H4, 0, 0,
	 0, 0, A5, B5, C5, D5, E5, F5, G5, H5, 0, 0,
	 0, 0, A6, B6, C6, D6, E6, F6, G6, H6, 0, 0,
	 0, 0, A7, B7, C7, D7, E7, F7, G7, H7, 0, 0,
	 0, 0, A8, B8, C8, D8, E8, F8, G8, H8, 0, 0,
	 0, 0, 0,  0,  0,  0,  0,  0,   0,  0, 0, 0,
	 0, 0, 0,  0,  0,  0,  0,  0,   0,  0, 0, 0);

   subtype Square_Color_Type is Integer range Frame .. Black; -- Empty, White and Black
   Square_Color : constant array (ChessBoard'Range) of Square_Color_Type :=
		    (
       Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame,
       Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame,
       Frame, Frame, White, Black, White, Black, White, Black, White, Black, Frame, Frame,
       Frame, Frame, Black, White, Black, White, Black, White, Black, White, Frame, Frame,
       Frame, Frame, White, Black, White, Black, White, Black, White, Black, Frame, Frame,
       Frame, Frame, Black, White, Black, White, Black, White, Black, White, Frame, Frame,
       Frame, Frame, White, Black, White, Black, White, Black, White, Black, Frame, Frame,
       Frame, Frame, Black, White, Black, White, Black, White, Black, White, Frame, Frame,
       Frame, Frame, White, Black, White, Black, White, Black, White, Black, Frame, Frame,
       Frame, Frame, Black, White, Black, White, Black, White, Black, White, Frame, Frame,
       Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame,
       Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame, Frame );


   function Evaluate return Integer;


   -- detect draws by insufficient material
   -- function does not detect ALL kind of draws,
   -- but It's close to do it
   function Draw_By_Insufficient_Material return Boolean;

private

   procedure Populate_Weak_Square_Board;

   function Evaluate_White_Pawn_Structure return Integer;
   function Evaluate_Black_Pawn_Structure return Integer;

   function Evaluate_White_King_Safety return Integer;
   function Evaluate_Black_King_Safety return Integer;

   function Evaluate_White_Material_Advantage (Score : in Integer) return Integer;
   function Evaluate_Black_Material_Advantage (Score : in Integer) return Integer;

   function Evaluate_White_Rooks return Integer;
   function Evaluate_Black_Rooks return Integer;

   function Evaluate_White_Piece_Development return Integer;
   function Evaluate_Black_Piece_Development return Integer;

   function Evaluate_White_Piece_Positional_Game return Integer;
   function Evaluate_Black_Piece_Positional_Game return Integer;

   function Evaluate_White_Mobility return Integer;
   function Evaluate_Black_Mobility return Integer;

   function Evaluate_White_Unprotected_Pieces return Integer;
   function Evaluate_Black_Unprotected_Pieces return Integer;


   -- calcola la distanza fra due caselle
   function Distance (From, To : in Integer) return Integer with Inline => True;

end ACEvaluate;
