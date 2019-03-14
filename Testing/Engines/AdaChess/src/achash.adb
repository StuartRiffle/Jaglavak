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


with Ada.Numerics.Discrete_Random;
with Interfaces; use Interfaces;
with ACChessBoard; use ACChessBoard;

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;


package body ACHash is


   ---------------------
   -- Initialize_Hash --
   ---------------------

   procedure Initialize_Hash is
      Skip : Hash_Type;
   begin
      Reset_Transposition_Table;
      Hash_Random.Reset (Seed);
      Skip := Hash_Random.Random (Seed); --yep, skip the first three randoms ;-)
      Skip := Hash_Random.Random (Seed);
      Skip := Hash_Random.Random (Seed);
      for Piece in White_Pawn .. Black_King loop
	 for Square in ChessBoard'Range loop
	    Hash_Pieces (Piece, Square) := Hash_Random.Random (Seed);
	 end loop;
      end loop;
      for I in ChessBoard'Range loop
	 Hash_En_Passant (I) := Hash_Random.Random (Seed);
      end loop;
      for I in Hash_Castle'Range loop
	 Hash_Castle (I) := Hash_Random.Random (Seed);
      end loop;
      Hash_Side := Hash_Random.Random (Seed);
   end Initialize_Hash;


   ----------------
   -- Reset_Hash --
   ----------------

   procedure Reset_Hash is
   begin
      Hash := 0;
   end Reset_Hash;


   -----------------
   -- Update_Hash --
   -----------------

   procedure Update_Hash is
   begin
      Reset_Hash;

      for I in White_Pieces'Range loop
	 exit when not Is_Piece (White_Pieces (I).Piece) and then not Is_Piece (Black_Pieces (I).Piece);
	 if Is_Piece (White_Pieces (I).Piece) then
	    Hash := Hash xor Hash_Pieces (White_Pieces (I).Piece, White_Pieces (I).Square);
	 end if;
	 if Is_Piece (Black_Pieces (I).Piece) then
	    Hash := Hash xor Hash_Pieces (Black_Pieces (I).Piece, Black_Pieces (I).Square);
	 end if;
      end loop;

      if Is_In_Board (En_Passant) then
	 Hash := Hash xor Hash_En_Passant (En_Passant);
      end if;
      if Castle.White_Kingside or else Castle.White_Queenside then
	 Hash := Hash xor Hash_Castle (Hash_Castle'First);
      end if;
      if Castle.Black_Kingside or else Castle.Black_Queenside then
	 Hash := Hash xor Hash_Castle (Hash_Castle'First + 2);
      end if;
      if Side_To_Move = Black then
	 Hash := Hash xor Hash_Side;
      end if;

   end Update_Hash;


   ------------------
   -- Generate_Key --
   ------------------

   function Generate_Key return Integer is
   begin
      return Integer (Hash mod Hash_Type (TT_Size));
   end Generate_Key;


   -----------------------------
   -- Set_Transposition_Score --
   -----------------------------

   procedure Set_Transposition_Score (Key, Depth, Score : in Integer; Hash : in Hash_Type; Best : in Boolean; Flag : in Value_Type) is
   begin
--        if Transposition_Table (Key).Hash = 0 or else (Transposition_Table (Ply, Key).Hash /= 0 and then Transposition_Table (Ply, Key).Best = False) then
	 Transposition_Table (Key).Hash := Hash;
	 Transposition_Table (Key).Depth := Depth + 1;
	 Transposition_Table (Key).Score := Score;
	 Transposition_Table (Key).Best := Best;
	 Transposition_Table (Key).Flag := Flag;
--        end if;
   end Set_Transposition_Score;


   --------------------------------
   -- Lookup_Transposition_Score --
   --------------------------------

   function Lookup_Transposition_Score (Key : in Integer) return Transposition_Table_Type is
   begin
      return Transposition_Table (Key);
   end Lookup_Transposition_Score;


   -------------------------------
   -- Reset_Transposition_Table --
   -------------------------------

   procedure Reset_Transposition_Table is
   begin
      for I in TT_Range loop
	 Transposition_Table (I).Hash := 0;
	 Transposition_Table (I).Depth := -1;
	 Transposition_Table (I).Score := 0;
	 Transposition_Table (I).Best := False;
	 Transposition_Table (I).Flag := Unknown;
      end loop;
   end Reset_Transposition_Table;


end ACHash;
