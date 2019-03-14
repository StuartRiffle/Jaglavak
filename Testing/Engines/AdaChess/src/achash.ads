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
with Ada.Numerics.Discrete_Random;
with Interfaces;	use Interfaces;

package ACHash is

   package Hash_Random is new Ada.Numerics.Discrete_Random (Hash_Type);
   use Hash_Random;

   -- Max_Number_Of_Piece_In_Game : constant Positive := 32;

    --Hash_Pieces    : array (Frame .. Black_King, ChessBoard'Range) of Hash_Type;
   Hash_Pieces    : array (White_Pawn .. Black_King, ChessBoard'Range) of Hash_Type;
   Hash_Side      : Hash_Type;
   Hash_En_Passant : array (ChessBoard'Range) of Hash_Type;
   Hash_Castle : array (1 .. 4) of Hash_Type;

   Seed           : Hash_Random.Generator;

   type Value_Type is (Unknown, Exact_Value, Upper_Bound, Lower_Bound);

   procedure Initialize_Hash;
   procedure Update_Hash with Inline => True;
   procedure Reset_Hash with Inline => True;

   type Transposition_Table_Type is
      record
	 Hash  : Hash_Type;
	 Depth : Integer;
	 Score : Integer;
	 Best  : Boolean;
	 Flag  : Value_Type;
      end record;

   subtype TT_Range is Integer range 0 .. 1123456;
   TT_Size : constant Integer := TT_Range'Last;


   Transposition_Table : array (TT_Range) of Transposition_Table_Type;

--     Transposition_Table := new Transposition_Table_Type

   function Generate_Key return Integer with Inline => True;
   procedure Set_Transposition_Score (Key, Depth, Score : in Integer; Hash : in Hash_Type; Best : in Boolean; Flag : in Value_Type);
   function Lookup_Transposition_Score (Key : in Integer) return Transposition_Table_Type with Inline => True;
   procedure Reset_Transposition_Table;
--     procedure Remove_Transposition_Table_Entry (Key : in Integer);

end ACHash;
