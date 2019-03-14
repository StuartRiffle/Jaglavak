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



with Ada.Calendar; use Ada.Calendar;
--  with ACChessBoard; use ACChessBoard;

package ACTimeManage is



   No_Time                    : constant Duration := 0.0; -- zero

   Thinking_Time_Exceeded     : exception;
   Need_Extra_Time            : Boolean; -- avoid exception raising
   Invalid_Time_Level         : exception;

   Max_Thinking_Depth         : Positive := 64;
   Max_Depth_Search : Integer := 63;


   function Use_Time_Management return Boolean with Inline => True;
   --function Set_Max_Thinking_Depth (Depth : in Ply_Type);


   -- Set the max time the engine can use to think
   -- for it's move. However, in some critical situation
   -- the engine can ask for extra time to perform
   -- some important operation.
   -- @param Seconds The time for each move, in seconds
   procedure Set_Time (Seconds : in Duration) with Inline => True;

   procedure Set_Time
     (Number_Of_Moves : in Integer;
      Total_Time      : in Duration;
      Increment       : in Duration) with Inline => True;

   -- set depth for each move.
   -- engine will search until this depth
   procedure Set_Depth (Depth : in Positive) with Inline => True;


   -- Retrieve the time that can be used while
   -- thinking on next move
   -- @return Duration The thinking time for next move
   function Get_Thinking_Time return Duration with Inline => True;

   -- Retrieve the time used to think from the beginning
   -- of the move until now
   -- @return Duration The thinked time
   function Get_Thinked_Time return Duration with Inline => True;

   -- Sometimes move search requires to grant
   -- extra time, for example any time the
   -- principal variation changes. This method
   -- give the time accordint to the time
   -- management type and game rules.
   -- There's no guarantee that any time the
   -- engine ask for more time it will be grant.
   procedure Ask_Extra_Time with Inline => True;
   procedure Ask_Extra_Time (Score_Diff : in Integer) with Inline => True;
   function Time_Left return Duration with Inline => True;


   procedure Start_Clock with Inline => True;
   procedure Stop_Clock with inline => True;

   function Time_Has_Up return Boolean with Inline => True;

private

   type Time_Management_Type is (Conventional, -- level 40 5 0 -- conventional game
				 Sudden_death, -- level 0 15 0
				 Exact_For_Move);  -- st 7

   Time_Increment : Duration; -- used as increment
   Time_Limit_Per_Move     : Duration; -- engine cannot overflow this thinking time
   Time_Limit              : Duration;


   Thinking_Time              : Duration; -- Thinking time for a single move
   Engine_Thinking_Time       : Time; -- how much has the engine effective thought about a move?

   Extra_Time                 : Duration;
   Extra_Time_Limit           : Duration;
   Extra_Time_Increment       : Duration := 0.1;
   Extra_Time_Limit_Fracion   : Duration := 2.0; -- thinking_time / Extra_time_Limit_Fraction


   Total_Thinking_Time : Duration;
   Time_Management     : Time_Management_Type;


   Engine_Moves_Counter : Integer := 0;
   Engine_Moves_Couter_Per_Iteration : Integer;
   Engine_Moves_Iteration : Integer := 0;
   Engine_Time_Limit_Per_Iteration : Duration;

end ACTimeManage;
