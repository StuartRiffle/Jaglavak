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
with Ada.Calendar; 	use Ada.Calendar;

package ACSearch is

   Current_Score : Integer;
   Check_Only_Recapture : Boolean;
   Iterative_Deep       : Integer;

   type Output_Type is (Standard, Xboard);
   for Output_Type'Size use 2;

   Ouptut_Mode : Output_Type := Standard;

   PV_Start_Time : Ada.Calendar.Time;
   PV_Engine_Thinking_Time : Duration;
   PV_Score : Float;
--     PV_Nodes_Per_Second : Integer;

   Nodes : Integer;


   Max_Ply : Integer;
   Search_Depth : Integer;

   Infinity : constant Integer := +1_000_000; -- the smallest one;
   Max_Quiescence_Ply          : constant Integer := 6;

   Aspiration_Window_Size : constant Integer := 33; -- tenth of a pawn or  one-third of a pawn are good choices
   Alpha_Window           : Integer;

   -- This is the principal variation line
   Principal_Variation : array (Ply_Type'Range, Ply_Type'Range) of Move_Type;
   -- This is the principal variation move at the first depths.
   -- if the move is the same at depth 4, 5and 6, then do the move
   -- Don't consider first 3 depth since there are not enough information
   Principal_Variation_Move : array (4 .. 6) of Move_Type;
   Enable_PV_Move           : Boolean; -- use the PV Move. Disable it for testing, debugging, benchmarking...

   -- other useful variables
   Following_Principal_Variation : Boolean;
   Principal_Variation_Depth : array (Ply_Type'Range) of Integer;
   Principal_Variation_Score     : Integer;
   -- use killer moves
   Killer_Moves                  : array (Ply_Type'Range, Ply_Type'Range) of Move_Type; -- moves that cause a beta cut-off
   Killer_Heuristic              : array (Ply_Type'Range) of Move_Type; -- this is the second variation
   Fail_Low_Heuristic            : array (ChessBoard'Range, ChessBoard'Range) of Integer; -- counter of fail low

   Last_Move                     : Move_Type;

   PV_Change : Integer; -- how many times has the PV changed? -- used for debug

   function Count_Repetitions return Natural with Inline => True;
   function Think (Max_Time : in Duration) return Move_Type;

   procedure Perft (Max_Depth : in Natural);
   procedure Perft_Search (Max_Depth : in Natural);

   procedure Divide (Max_Depth : in Natural);
   procedure Divide_Search (Max_Depth : in Natural);

   function Principal_Variation_Search (Max_Depth : in Natural; Alpha, Beta : in Integer) return Integer;
   function Zero_Window_Search (Max_Depth : in Natural; Alpha, Beta : in Integer) return Integer;

   function Quiescence (Alpha, Beta : in Integer) return Integer;


   procedure Sort_Moves;
   procedure Sort (Index : in Integer);
   procedure Quick_Sort (From, To : in Integer);

   procedure Print_Principal_Variation (Score : in Integer);

end ACSearch;
