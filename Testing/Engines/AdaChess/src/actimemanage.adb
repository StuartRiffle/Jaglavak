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


with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
with Ada.Float_Text_IO; use Ada.Float_Text_IO;
with Ada.Calendar; 	use Ada.Calendar;

package body ACTimeManage is


   -------------------------
   -- Use_Time_Management --
   -------------------------

   function Use_Time_Management return Boolean is
   begin
      return Max_Thinking_Depth = 64;
   end Use_Time_Management;


   --------------
   -- Set_Time --
   --------------

   procedure Set_Time (Seconds : in Duration) is
   begin
      Time_Limit_Per_Move := Seconds;
      Extra_Time_Limit := Seconds / Extra_Time_Limit_Fracion;
      Extra_Time := 0.0;
      Time_Management := Exact_For_Move;
      Time_Limit := 1000000.0;
   end Set_Time;


   --------------
   -- Set_Time --
   --------------

   -- this function answer to the xboard command
   -- level 40 5 0
   -- 40 moves for each time loop
   -- 5 is the time for the 40 moves, in minutes
   -- 0 is the time increment in seconds
   procedure Set_Time
     (Number_Of_Moves : in Integer;
      Total_Time      : in Duration;
      Increment       : in Duration) is
   begin
--        Put ("Time management set to: ");
      if Number_Of_Moves = 0 and then Increment = 0.0 then
	 Time_Management := Sudden_Death;
	 Time_Increment := Increment;
	 Time_Limit := Total_Time * 60;
	 Time_Limit_Per_Move := Time_Limit / 40.0;
	 Extra_Time_Limit := Time_Limit_Per_Move / Extra_Time_Limit_Fracion;
	 Extra_Time := 0.0;
	 Engine_Moves_Counter := 0;
--  	 Put_Line ("Sudden death");
      elsif Number_Of_Moves = 0 and then Increment > 0.0 then
	 Time_Management := Sudden_Death;
	 Time_Increment := Increment;
	 Time_Limit := Total_Time * 60;
	 Time_Limit_Per_Move := Time_Limit / 40.0;
	 Extra_Time_Limit := Time_Limit_Per_Move / Extra_Time_Limit_Fracion;
	 Extra_Time := 0.0;
	 Engine_Moves_Counter := 0;

--  	 Put_Line ("Sudden death");
      elsif Number_Of_Moves > 0 and then Increment = 0.0 then
	 Time_Management := Conventional;
	 Time_Increment := Increment;
	 Engine_Moves_Couter_Per_Iteration := Number_Of_Moves;
	 Engine_Moves_Counter := 0;
	 Engine_Moves_Iteration := 0;
	 Time_Limit := Total_Time * 60;
	 Engine_Time_Limit_Per_Iteration := Time_Limit;
--  	 Put_Line ("Conventional");
      elsif Number_Of_Moves > 0  and then Increment > 0.0 then
	 Time_Management := Conventional;
	 Time_Increment := Increment;
	 Engine_Moves_Couter_Per_Iteration := Number_Of_Moves;
	 Engine_Moves_Counter := 0;
	 Engine_Moves_Iteration := 0;
	 Time_Limit := Total_Time * 60;
	 Engine_Time_Limit_Per_Iteration := Time_Limit;
--  	 Put_Line ("Conventional");
      else
	 raise Invalid_Time_Level;
      end if;

   end Set_Time;


   ---------------
   -- Set_Depth --
   ---------------

   procedure Set_Depth (Depth : in Positive) is
   begin
      Max_Depth_Search := Depth;
      Time_Limit := 1000000.0; -- should be enough ;-)
      Time_Management := Exact_For_Move;
      Time_Limit_Per_Move := Time_Limit;
   end Set_Depth;


   -----------------------
   -- Get_Thinking_Time --
   -----------------------

   function Get_Thinking_Time return Duration is
      Moves_Left : Integer;
   begin
      case Time_Management is
	 when Conventional =>
	    -- calc how much time to spend for this move
	    -- the calc is very easy: time left uniformly
     -- divided for the number of moves left
	    Moves_Left := Engine_Moves_Couter_Per_Iteration - Engine_Moves_Counter;

	    Thinking_Time := (Time_Limit / Moves_Left);
	    Thinking_Time := Thinking_Time + Time_Increment;
	    Time_Limit_Per_Move := Time_Limit;
	    --  	    Put_Line ("Moves left: " & Integer'Image (Moves_Left));
	    --  	    Put ("Time to think for this move: ");
	    --  	    Put ( Item => Float (Thinking_Time), Fore => 3, Aft => 2, Exp => 0);
	    --  	    New_Line;
	 when Sudden_Death =>
	    -- calc how much time to spend for this move
	    if Engine_Moves_Counter < 20 then
	       Thinking_Time := Time_Limit_Per_Move;
	    else
	       Thinking_Time := Time_Limit / 20.0;
	    end if;
	    Thinking_Time := Thinking_Time + Time_Increment;
	    Put ("Time limit: ");
	    Put (Item => Float (Thinking_Time), Fore => 3, Aft => 2, Exp => 0);
	    New_Line;
	 when Exact_For_Move =>
	    Thinking_Time := Time_Limit_Per_Move;
	    -- calc how much time to spend for this move
      end case;

      -- don't use the whole time because some operations
      -- like the reset of the history heuristic and other things
      -- takes some time. So grant an extra time
      -- to avoid to go in an overflow when play with
      -- external time management like in Xboard or Arena
      return Thinking_Time - 0.25;
   end Get_Thinking_Time;


   -----------------------
   -- Get_Thinked_Time --
   -----------------------

   function Get_Thinked_Time return Duration is
   begin
      return Clock - Engine_Thinking_Time;
   end Get_Thinked_Time;


   --------------------
   -- Ask_Extra_Time --
   --------------------

   procedure Ask_Extra_Time is
   begin
      case Time_Management is
	 when Exact_For_Move => null; -- can't give extra time
	 when Conventional | Sudden_Death =>
	    if Extra_Time + Extra_Time_Increment <= Extra_Time_Limit then
	       Extra_Time := Extra_Time + Extra_Time_Increment;
	    else
	       Extra_Time := Extra_Time_Limit;
	    end if;
      end case;
   end Ask_Extra_Time;


   --------------------
   -- Ask_Extra_Time --
   --------------------

   procedure Ask_Extra_Time (Score_Diff : in Integer) is
      Current_Extra_Time_Increment : Duration;
   begin
      Current_Extra_Time_Increment := Extra_Time_Increment;
      Extra_Time_Increment := Duration (abs (Score_Diff / 1000));
      Ask_Extra_Time;
      Extra_Time_Increment := Current_Extra_Time_Increment;
   end Ask_Extra_Time;


   ---------------
   -- Time_Left --
   ---------------

   function Time_Left return Duration is
   begin
      return Time_Limit - Get_Thinked_Time;
   end Time_Left;


   -----------------
   -- Start_Clock --
   -----------------

   procedure Start_Clock is
   begin
      Extra_Time := 0.0;
      Engine_Thinking_Time := Clock;
      Extra_Time_Limit := Thinking_Time / 10.0;
      Extra_Time := 0.0;
   end Start_Clock;


   ----------------
   -- Stop_Clock --
   ----------------

   procedure Stop_Clock is
   begin
      case Time_Management is
	 when Exact_For_Move => null;
	 when Conventional =>
	    Engine_Moves_Counter := Engine_Moves_Counter + 1;
	    if Engine_Moves_Counter >= Engine_Moves_Couter_Per_Iteration then
	       Engine_Moves_Iteration := Engine_Moves_Iteration + 1;
	       Engine_Moves_Counter := 0;
	       Time_Limit := Engine_Time_Limit_Per_Iteration;
	    end if;
	 when Sudden_Death => Engine_Moves_Counter := Engine_Moves_Counter + 1;
      end case;
      Time_Limit := Time_Limit - Get_Thinked_Time + Time_Increment; -- take back the increment!
      if Time_Limit < 0.0 then
	 Time_Limit := 0.0;
      end if;
   end Stop_Clock;


   -----------------
   -- Time_Has_Up --
   -----------------

   function Time_Has_Up return Boolean is
      Thinked_Time : Duration;
   begin
      Thinked_Time := Get_Thinked_Time;
      return Thinked_Time >= Thinking_Time + Extra_Time or else Thinked_Time > Time_Limit_Per_Move;
   end Time_Has_Up;


end ACTimeManage;
