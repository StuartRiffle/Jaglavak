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
with Ada.Text_IO; use Ada.Text_IO;
--  with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
--  with Interfaces; use Interfaces;


package body ACFen is


   --------------
   -- Fen_Init --
   --------------

   procedure Fen_Init is
   begin
      Initialize;
      ChessBoard := (others => Frame);
      Castle := (False, False, False, False);
      En_Passant := 0;
   end Fen_Init;


   ---------------------
   -- Fen_Load_Pieces --
   ---------------------

   procedure Fen_Load_Pieces (Fen : in String) is
      Item                   : Character;
      Next, Castle_Count     : Natural := 0;
      Square, Sq             : Integer := 1;
      Board                  : array (1 .. 64) of Integer :=
	(A8, B8, C8, D8, E8, F8, G8, H8,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A1, B1, C1, D1, E1, F1, G1, H1 );
   begin

      for I in Fen'Range loop
	 Item := Fen (I);
	 Square := Board (Sq);
	 case Item is
	    when 'P' =>
	       ChessBoard (Square) := White_Pawn;
	    when 'N' =>
	       ChessBoard (Square) := White_Knight;
	    when 'B' =>
	       ChessBoard (Square) := White_Bishop;
	    when 'R' =>
	       ChessBoard (Square) := White_Rook;
	    when 'Q' =>
	       ChessBoard (Square) := White_Queen;
	    when 'K' =>
	       ChessBoard (Square) := White_King;
	       White_King_Position := Square;
	    when 'p' =>
	       ChessBoard (Square) := Black_Pawn;
	    when 'n' =>
	       ChessBoard (Square) := Black_Knight;
	    when 'b' =>
	       ChessBoard (Square) := Black_Bishop;
	    when 'r' =>
	       ChessBoard (Square) := Black_Rook;
	    when 'q' =>
	       ChessBoard (Square) := Black_Queen;
	    when 'k' =>
	       ChessBoard (Square) := Black_King;
	       Black_King_Position := Square;
	    when '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' =>
	       for J in Sq .. Sq + Character'Pos (Item) - Character'Pos ('1') loop
		  Square := Board (J);
		  ChessBoard (Square) := Empty;
	       end loop;
	       Sq := Sq + Character'Pos (Item) - Character'Pos ('1');
	    when others => Sq := Sq - 1; -- it come here when '/' occur. since we update sq later, here we ensure that sq will not skip any squares
	 end case;
	 if Sq < Board'Last then
	    Sq := Sq + 1;
	 end if;
	 Next := Next + 1;
	 exit when Item = ' '; -- board set-up ends here
      end loop;
         -- update pieces
      White_Pieces := (others => No_Piece);
      White_Pieces_Counter := 0;
      Black_Pieces := (others => No_Piece);
      Black_Pieces_Counter := 0;
      for I in ChessBoard'Range loop
	 if Is_Piece (ChessBoard (I)) then
	    if Is_White (ChessBoard (I)) then
	       Add_White_Piece (I, ChessBoard (I));
	    elsif Is_Black (ChessBoard (I)) then
	       Add_Black_Piece (I, ChessBoard (I));
	    end if;
	 end if;
      end loop;

      Align_Piece_Table;

   end Fen_Load_Pieces;


   ---------------------------
   -- Fen_Load_Side_To_Move --
   ---------------------------

   procedure Fen_Load_Side_To_Move (Fen : in String) is
   begin
      if Fen = "w" then
	 Side_To_Move := White;
      else
	 Side_To_Move := Black;
      end if;
   end Fen_Load_Side_To_Move;


   ---------------------------
   -- Fen_Load_Castle_Flags --
   ---------------------------

   procedure Fen_Load_Castle_Flags (Fen : in String) is
      Item : Character;
   begin
      for I in Fen'Range loop
	 Item := Fen (I);
	 case Item is
	    when 'K' =>
	       Castle.White_Kingside := True;
	    when 'Q' =>
	       Castle.White_Queenside := True;
	    when 'k' =>
	       Castle.Black_Kingside := True;
	    when 'q' =>
	       Castle.Black_Queenside := True;
	    when '-' => null; -- no castle available
	    when others => null; -- should never come here.
	 end case;
      end loop;
   end Fen_Load_Castle_Flags;


   ---------------------------
   -- Fen_Load_En_Passant --
   ---------------------------

   procedure Fen_Load_En_Passant (Fen : in String) is
   begin
      if Fen'Length = 2 then
	 En_Passant := 2 + Character'Pos (Fen (Fen'First)) - Character'Pos ('a') +
	   12 * ( 10 - (Character'Pos (Fen (Fen'First + 1)) - Character'Pos ('0') ));
      end if;
   end Fen_Load_En_Passant;


   ------------------------------
   -- Fen_Load_Half_Move_Clock --
   ------------------------------

   procedure Fen_Load_Half_Move_Clock (Fen : in String) is
   begin
      null; -- skip clock counter, we don't need it
   end Fen_Load_Half_Move_Clock;


   -------------------------------
   -- Fen_Load_Fullmove_Counter --
   -------------------------------

   procedure Fen_Load_Fullmove_Counter (Fen : in String) is
   begin
      History_Ply := Integer'Value (Fen);
      History_Started_At := History_Ply; -- update history starting point.
   end Fen_Load_Fullmove_Counter;


end ACFen;
