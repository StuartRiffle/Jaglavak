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
with ACEvaluate; use ACEvaluate;
with ACTimeManage; use ACTimeManage;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
with Ada.Float_Text_IO; use Ada.Float_Text_IO;
with Ada.Calendar; use Ada.Calendar;
with Interfaces; use Interfaces;
with ACHash; 	use ACHash;
with ACBook;	use ACBook;


package body ACSearch is


   ----------------
   -- Sort_Moves --
   ----------------

   procedure Sort_Moves is
   begin
      Following_Principal_Variation := False;

      for I in Ply_Type'First .. Ply_Type'First + Moves_Counter (Ply) - 1 loop
	 if Moves_List (Ply, I) = Principal_Variation (1, Ply) then
	    Following_Principal_Variation := True;
	    Moves_List (Ply, I).Score := 5000;
	 elsif Moves_List (Ply, I) = Killer_Heuristic_1 (Ply) then
	    Moves_List (Ply, I).Score := 1000;
	 elsif Moves_List (Ply, I) = Killer_Heuristic_2 (Ply) then
	    Moves_List (Ply, I).Score := 500;
	 elsif Moves_List (Ply, I) = Killer_Heuristic_3 (Ply) then
	    Moves_List (Ply, I).Score := 250;
	 else
	    Assign_Score (Moves_List (Ply, I));
	 end if;
      end loop;

      Quick_Sort (Ply_Type'First, Moves_Counter (Ply));
   end Sort_Moves;


   ----------
   -- Sort --
   ----------

   procedure Sort (Index : in Integer) is
      Best_Score, Best_Index : Integer;
      Move                   : Move_Type;
   begin
      Best_Score := -1000000;
      Best_Index := Index;
      for I in Index .. Moves_Counter (Ply) loop
	 if Moves_List (Ply, I).Score > Best_Score then
	    Best_Score := Moves_List (Ply, I).Score;
	    Best_Index := I;
	 end if;
      end loop;
      -- swap moves
      Move := Moves_List (Ply, Index);
      Moves_List (Ply, Index) := Moves_List (Ply, Best_Index);
      Moves_List (Ply, Best_Index) := Move;
   end Sort;

   ----------------
   -- Quick_Sort --
   ----------------

   procedure Quick_Sort (From, To : in Integer) is
      Pivot, Pivot_Score : Integer;
      Move               : Move_Type;
   begin
      if To <= From then
	 return;
      end if;

      Pivot := From;
      Pivot_Score := Moves_List (Ply, From).Score;
      for I in From + 1 .. To loop
	 if Moves_List (Ply, I).Score > Pivot_Score then
	    Pivot := Pivot + 1;
	    Move := Moves_List (Ply, Pivot);
	    Moves_List (Ply, Pivot) := Moves_List (Ply, I);
	    Moves_List (Ply, I) := Move;
	 end if;
      end loop;

      -- take back the pivot
      if Pivot /= From then
	 Move := Moves_List (Ply, From);
	 Moves_List (Ply, From) := Moves_List (Ply, Pivot);
	 Moves_List (Ply, Pivot) := Move;
      end if;

      -- sort on left and right
      Quick_Sort (From, Pivot - 1);
      Quick_Sort (Pivot + 1, To);

   end Quick_Sort;


   -----------------------
   -- Count_Repetitions --
   -----------------------

   function Count_Repetitions return Natural is
      package Hash_IO is new Ada.Text_IO.Modular_IO (Unsigned_64);
      use Hash_IO;
      Repetitions : Natural := 1; -- this position Happens the first time NOW
   begin
      if History_Ply > 6 then
	 for I in reverse 6 .. History_Ply - 1 loop
	    if History_Moves (I).Hash = Hash then
	       Repetitions := Repetitions + 1;
	    end if;
	    exit when Repetitions = 3;
	 end loop;
      end if;
      return Repetitions;
   end Count_Repetitions;


   -----------
   -- Think --
   -----------

   function Think (Max_Time : in Duration) return Move_Type is
      Move                  : Move_Type;
      Score                 : Integer;
      Alpha, Beta           : Integer;
      Use_Aspiration_Window_Search : Boolean;
   begin

      Generate_Moves;

      -- Try move from book first
      Move := Book_Move;
      if Move /= No_Move then
	 Put_Line ("Book move");
	 return Move;
      end if;

      if Moves_Counter (Ply) = 1 then -- play that move, no need to search
	 Put_Line ("I'm forced to play the only 1 legal move!");
	 return Moves_List (Ply, 1);
      end if;

      if Moves_Counter (Ply) = 0 then
	 return No_Move; -- checkmate?
      end if;


      for P in Ply_Type'Range loop
	 for I in ChessBoard'Range loop
	    for J in ChessBoard'Range loop
	       if p <= 2 then
		  History_Heuristic (P, I, J) := 0;
	       else
		  History_Heuristic (P, I, J) := History_Heuristic (P - 2, I, J);
	       end if;
	    end loop;
	 end loop;
      end loop;

      -- reset principal variation
      for I in Ply_Type'Range loop
	 for J in Ply_Type'Range loop
	    Principal_Variation (I, J) := No_Move;
	 end loop;
	 Principal_Variation_Depth (I) := 0;
      end loop;

      for I in Principal_Variation_Move'Range loop
	 Principal_Variation_Move (I) := No_Move;
      end loop;

      Ply := Ply_Type'First;

      Nodes := 0;
      Score := 0;

      PV_Start_Time := Clock;

      Principal_Variation_Score := 0;

      Following_Principal_Variation := False;
      -- iterative deeping search
      Alpha := -Infinity;
      Beta := +Infinity;
      Max_Ply := 0;
      Reset_Transposition_Table;
      Search_Depth := 1; -- trick: on the 1st pass, look for quiescence!
      Alpha_Window := -Infinity;

      PV_Change := 0;

      Current_Score := Evaluate;

      if Ouptut_Mode = Standard then
	 Put_Line ("Ply  Time      Nodes     Score Principal Variation");
--        elsif Ouptut_Mode = Xboard then
--  	 Put_Line ("Ply     Nodes     Score Principal Variation");
      end if;

      Need_Extra_Time := False;
      Use_Aspiration_Window_Search := True;

      for I in Ply_Type'Range loop

	 -- dont use aspiration window on deep search.
	 -- because it can require too much time
	 if Search_Depth > 5 then
	    Use_Aspiration_Window_Search := False;
	 end if;

	 Score := Principal_Variation_Search (Search_Depth, Alpha, Beta);
	 if Score <= Alpha or else Score >= Beta then
	    Alpha := -Infinity;
	    Beta := +Infinity;
	    Ask_Extra_Time;
	    Need_Extra_Time := True; -- prevent PV search to stop search for time
	 else
	    Need_Extra_Time := False;

	    -- sometimes the move is "forced" and there's no need
	    -- to search deeper. This is when the PV doesn't change
	    -- at first depth. Here I will pick the move if it is the
	    -- same at depth described in Principal_Variation_Move.
	    -- good choice as Principal_Variation_Move'Range is 4 .. 6
	    if Enable_PV_Move and then Search_Depth in Principal_Variation_Move'Range then
	       Principal_Variation_Move (Search_Depth) := Principal_Variation (1, 1);
	       if Principal_Variation_Move (Principal_Variation_Move'First) = Principal_Variation_Move (Principal_Variation_Move'First + 1)
		 and then Principal_Variation_Move (Principal_Variation_Move'First) = Principal_Variation_Move (Principal_Variation_Move'First + 2) then
		  New_Line;
		  return Principal_Variation_Move (Principal_Variation_Move'First);
	       end if;
	    end if;

	    if Use_Aspiration_Window_Search then
	       Alpha := Score - Aspiration_Window_Size;
	       Beta := Score + Aspiration_Window_Size;
	    end if;
	    Search_Depth := Search_Depth + 1;

	    Principal_Variation_Score := Score;
	    Print_Principal_Variation (Score);

	    PV_Engine_Thinking_Time := Clock - PV_Start_Time;

	 end if;

	 exit when Principal_Variation_Score > 990_900 or else Principal_Variation_Score < -990_900;
	 exit when Search_Depth > Max_Depth_Search;
      end loop;

      New_Line;
      return Principal_Variation (1, 1);

   exception
      when Time_Control : Thinking_Time_Exceeded =>

	 New_Line;
	 Put ("PV: ");
	 for I in 1 .. Principal_Variation_Depth (1) - 1 loop
	    Move := Principal_Variation (1, I);
	    Echo (Move);
	    Put (" ");
	 end loop;
	 New_Line;

	 Undo (Ply - 1);

	 return Principal_Variation (1, 1);
   end Think;



   --------------------------------
   -- Principal_Variation_Search --
   --------------------------------

   function Principal_Variation_Search (Max_Depth : in Natural; Alpha, Beta : in Integer) return Integer is
      Depth                                   : Natural;
      Current_Tree_Search_Depth               : Natural;
      Move                                    : Move_Type;
      Score                                   : Integer;
      Alpha_Score                             : Integer;
      Beta_Score                              : Integer;
      Cut_Offs                                : Integer;
      Key                                     : Integer;
      TT_Entry                                : Transposition_Table_Type;
      Current_Hash                            : Hash_Type;
      Depth_Extended                          : Boolean;
      Current_Depth                           : Integer;
      Moves_Searched                          : Integer := 0;
      Killer_Moves_Searched                   : Integer := 0;
   begin

      Depth_Extended := False;
      Depth := Max_Depth;
      Current_Tree_Search_Depth := Max_Depth;
      Principal_Variation_Depth (Ply) := Ply;

      if Ply > Max_Ply then
	 Max_Ply := Ply;
      end if;

      Alpha_Score := Alpha;
      Beta_Score := Beta;

      if Has_King_In_Check (Side_To_Move) then
	 Depth := Depth + 1;
	 Depth_Extended := True;
      end if;

      if Depth <= 0 or else Ply > Max_Depth_Search then
	 return Quiescence (Alpha, Beta);
      end if;

      if not Need_Extra_Time and then Time_Has_Up then
	 raise Thinking_Time_Exceeded;
      end if;

      Generate_Moves;
      Sort_Moves;

      if Moves_Counter (Ply) = 1 and then not Depth_Extended then
	 Depth := Depth + 1;
	 Depth_Extended := True;
      end if;

      if Count_Repetitions = 3 then
	 return Draw;
      end if;

      if Fifty >= 100 then
	 return Draw;
      end if;

      Nodes := Nodes + 1;

      Cut_Offs := 0;

      Current_Depth := Depth;

      for I in Ply_Type'First .. Moves_Counter (Ply) loop
	 exit when not Is_Valid (Moves_List (Ply, I));
--  	 exit when Depth > Depth_Cut_Offs and then Moves_Searched > Full_Depth_Moves;

	 Move := Moves_List (Ply, I);

	 Current_Hash := Hash;

	 if Play (Move) then

	    if not Is_Capture(Move) then
	       Moves_Searched := Moves_Searched + 1;
	    end if;

	    Key := Generate_Key;
	    TT_Entry := Lookup_Transposition_Score (Key);

	    if TT_Entry.Hash = Current_Hash and then TT_Entry.Depth < Ply then
	       if TT_Entry.Flag = Exact_Value then
--  		  Score := -TT_Entry.Score;
		  Score := -Principal_Variation_Search (Depth - 1, -Beta_Score, -Alpha_Score);
		  Undo;
		  return Score;
		  elsif TT_Entry.Flag = Upper_Bound and then TT_Entry.Score <= Alpha_Score then
		     Undo;
		     return Alpha_Score;
		  elsif TT_Entry.Flag = Lower_Bound and then TT_Entry.Score >= Beta_Score then
		     Undo;
		     return Beta_Score;
		  else
		     -- no information about TT_Entry? Then it's better to serach again
		     -- because something's got wrong, maybe a collision!
		     Score := -Principal_Variation_Search (Depth - 1, -Beta_Score, -Alpha_Score);
		     Undo;
	       end if;

	    else
	       Score := -Principal_Variation_Search (Depth - 1, -Beta_Score, -Alpha_Score);
	       Undo;

	    end if;

	 end if;

	    -- any time we found a score better than the previous one we have
	    -- found a better move. Sometimes this score is so good that we can
	    -- perform a tree cut-off. In other cases, the only thing to do is
	    -- an update of the Principal Variation.
	    if Score > Alpha_Score then

	    -- Update the cut-offs counter so on the next moves
	    -- we just have to search for better move instead of
	    -- performing a full-search on the tree
	    Cut_Offs := Cut_Offs + 1;

	    -- now we have a real score that we can use for move ordering
	    -- on the next iteration over move search. Of course, the deeper
	    -- is the search the most accurate is the heuristic.
	    -- However, the evaluation function needs to be accurate too

	    if not Is_Capture (Move) then
	       History_Heuristic (Ply, Move.From, Move.To) := Cut_Offs;
	    end if;

	    if Score >= Beta_Score then

	       -- update the transposition table data with this
	       -- value: We found a beta cut-off so set this
	       -- value as Lower Bound
	       Set_Transposition_Score (Key   => Key,
				 Depth => Ply,
				 Score => Beta_Score,
				 Hash  => Hash,
				 Best  => False,
				 Flag  => Lower_Bound);

	       Update_Killer_Moves (Move  => Move, Score => Score);

	       return Beta_Score;

	    end if;

	       Alpha_Score := Score;

	       -- update the transposition table data with this
	       -- value: We found a cut-off so set this
	       -- value as exact score found
	       Set_Transposition_Score (Key   => Key,
				 Depth => Ply,
				 Score => Score,
				 Hash  => Hash,
				 Best  => True,
				 Flag  => Exact_Value);


	       -- update Principal Variation
	       Principal_Variation (Ply, Ply) := Move;
	       for J in Ply + 1 .. Principal_Variation_Depth (Ply + 1) loop
		  Principal_Variation (Ply, J) := Principal_Variation (Ply + 1, J);
	       end loop;
	       Principal_Variation_Depth (Ply) := Principal_Variation_Depth (Ply + 1);
	       Principal_Variation_Score := Score;

--  	    Print_Principal_Variation (Score);
--  	    delay 0.1;

	    -- A new Principal variation found. Ask for extra
	    -- time to think deeper. Pass the score difference to find
	    -- if it is needed a lot of extra time or just a litte
--  	    Ask_Extra_Time (Score - Current_Score);
	    Ask_Extra_Time;

	    PV_Change := PV_Change + 1;

	 else -- score < alpha

	       -- update the transposition table data with this
	       -- value: We found a score that not caoused a cut-off
	       -- so Set This value as Upper Bound
	       Set_Transposition_Score (Key   => Key,
				 Depth => Ply,
				 Score => Alpha_Score,
				 Hash  => Hash,
				 Best  => False,
				 Flag  => Upper_Bound);
	    end if;

      end loop;

      -- do we end the game?
      if Moves_Counter (Ply) = 0 then
	 if Has_King_In_Check (Side_To_Move) then
	    return -CheckMate + Ply;
	 else
	    return Draw;
	 end if;
      end if;

      return Alpha_Score;

   end Principal_Variation_Search;


   ------------------------
   -- Zero_Window_Search --
   ------------------------

   function Zero_Window_Search (Max_Depth : in Natural; Alpha, Beta : in Integer) return Integer is
      Depth                                   : Natural;
      Move                                    : Move_Type;
      Score                                   : Integer;
      Alpha_Score                             : Integer;
      Beta_Score                              : Integer;
      Key                                     : Integer;
      TT_Entry                                : Transposition_Table_Type;
      Current_Hash                            : Hash_Type;
      Moves_Searched                          : Integer := 0;
      Killer_Moves_Searched                   : Integer := 0;
      Depth_Extended                          : Boolean := False;
   begin

      Depth := Max_Depth;
      Principal_Variation_Depth (Ply) := Ply;

      if Has_King_In_Check (Side_To_Move) then
	 Depth := Depth + 1;
	 Depth_Extended := True;
      end if;

      if Time_Has_Up then
	 raise Thinking_Time_Exceeded;
      end if;

      Alpha_Score := Alpha;
      Beta_Score := Beta;

      if Depth <= 0 then
	 return Quiescence (Alpha, Beta);
      end if;


      Generate_Moves;
      Sort_Moves;

      if Moves_Counter (Ply) = 1 and then Depth_Extended = False then
	 Depth := Depth + 1;
	 Depth_Extended := True;
      end if;


      if Count_Repetitions = 3 then
	 return Draw;
      end if;

      if Fifty >= 100 then
	 return Draw;
      end if;

      Nodes := Nodes + 1;


      for I in Ply_Type'First .. Moves_Counter (Ply) loop
	 exit when not Is_Valid (Moves_List (Ply, I));
--  	 exit when  > Stop_Searching (Ply);

	 Move := Moves_List (Ply, I);

	 Current_Hash := Hash;

	 if Play (Move) then

	    Moves_Searched := Moves_Searched + 1;
	    Key := Generate_Key;
	    TT_Entry := Lookup_Transposition_Score (Key);

	    if TT_Entry.Hash = Current_Hash and then TT_Entry.Depth < Ply then
	       if TT_Entry.Flag = Exact_Value then
--  		  Score := -TT_Entry.Score;
		  Score :=  -Zero_Window_Search (Depth - 1, -Beta_Score, -Alpha_Score);
		  Undo;
		  return Score;
	       elsif TT_Entry.Flag = Upper_Bound and then TT_Entry.Score <= Alpha_Score then
		  Undo;
		  return Alpha_Score;
	       elsif TT_Entry.Flag = Lower_Bound and then TT_Entry.Score >= Beta_Score then
		  Undo;
		  return Beta_Score;
	       else
		  -- no information about TT_Entry? Then it's better to serach again
		  -- because something's got wrong, maybe a collision!
		  Score :=  -Zero_Window_Search (Depth - 1, -Beta_Score, -Alpha_Score);
		  Undo;
	       end if;
	    else
	       Score := -Zero_Window_Search (Depth - 1, -Beta_Score, -Alpha_Score);
	       Undo;
	    end if;
	 end if;

	 if Score > Alpha_Score then
	   return Score;
	 end if;

      end loop;

      if Moves_Counter (Ply) = 0 then
	 if Has_King_In_Check (Side_To_Move) then
	    return -CheckMate + Ply;
	 else
	    return Draw;
	 end if;
      end if;

      return Alpha_Score;

   end Zero_Window_Search;




   ----------------
   -- Quiescence --
   ----------------

   function Quiescence (Alpha, Beta : in Integer) return Integer is
      Move                                : Move_Type;
      Score                               : Integer;
      Alpha_Score                         : Integer;
      Beta_Score                          : Integer;
   begin

      Principal_Variation_Depth (Ply) := Ply;

      Alpha_Score := Alpha;
      Beta_Score := Beta;

      Score := Evaluate;

      if Score >= Beta then
--  	 Update_Killer_Moves (Move  => Last_Move_Played, Score => Score);
	 return Beta;
      end if;


      if Score > Alpha_Score then
	 Alpha_Score := Score;
      end if;

      Nodes := Nodes + 1;

      Generate_Capture_Moves;
--        Generate_Moves;
      Sort_Moves;



      for I in Ply_Type'First .. Moves_Counter (Ply) loop
	 -- exit when not Is_Valid (Moves_List (Ply, I));
	 Move := Moves_List (Ply, I);
	 if Is_Piece (Move.Captured) or else Is_Piece (Move.Promotion) then -- or else Has_King_In_Check (Side_To_Move) then

	       if Play (Move) then

	       Score := -Quiescence (-Beta_Score, -Alpha_Score);
	       Undo;

	       if Score > Alpha_Score then

		  if Score >= Beta_Score then
		     Update_Killer_Moves (Move => Move, Score => Score);
		     return Beta_Score;
		  end if;
		  Alpha_Score := Score;

		  -- update Principal Variation
		  Principal_Variation (Ply, Ply) := Move;
		  for J in Ply + 1 .. Principal_Variation_Depth (Ply + 1) loop
		     Principal_Variation (Ply, J) := Principal_Variation (Ply + 1, J);
		  end loop;
		  Principal_Variation_Depth (Ply) := Principal_Variation_Depth (Ply + 1);
		  Principal_Variation_Score := Score;
--
--  		  Print_Principal_Variation (Score);
--  		  delay 0.1;

	       end if;

	    end if;
	 end if;

      end loop;

      return Alpha_Score;

   end Quiescence;


   -----------
   -- Perft --
   -----------

   procedure Perft (Max_Depth : in Natural) is
      Depth                 : Natural := Max_Depth;
      Start_Time, Stop_Time : Ada.Calendar.Time;
   begin
      Start_Time := Clock;
      Put_Line ("Perft   Nodes   Time");
      for I in 1 .. Max_Depth loop
	 Nodes := 0; -- reset nodes at each call: nodes represent legal moves in perft
	 Perft_Search (I);
	 Stop_Time := Clock;
	 Put (I, 0);
	 Put (" ");
	 Put (Nodes);
	 Put (" ");
	 Put (Item => Float (Stop_Time - Start_Time), Fore => 3, Aft => 2, Exp => 0);
	 New_Line;
      end loop;
      New_Line;
   end Perft;


   ------------------
   -- Perft_Search --
   ------------------

   procedure Perft_Search (Max_Depth : in Natural) is
   begin
      if Max_Depth > 0 then
	 Generate_Moves;
	 for I in Ply_Type'Range loop
	    exit when not Is_Valid (Moves_List (Ply, I));
	    if Play (Moves_List (Ply, I)) then
	       if Max_Depth = 1 then
		  Nodes := Nodes + 1;
	       end if;
	       Perft_Search (Max_Depth => Max_Depth - 1);
	       Undo;
	    end if;
	 end loop;
      end if;
   end Perft_Search;


   ------------
   -- Divide --
   ------------

   procedure Divide (Max_Depth : in Natural) is
      Depth                 : Natural := Max_Depth;
      Start_Time, Stop_Time : Ada.Calendar.Time;
   begin
      Start_Time := Clock;
      Put_Line ("Perft   Nodes   Time");
      for I in 1 .. Max_Depth loop
	 Nodes := 0; -- reset nodes at each call: nodes represent legal moves in perft
	 Divide_Search (I);
	 Stop_Time := Clock;
	 Put (I, 0);
	 Put (" ");
	 Put (Nodes);
	 Put (" ");
	 Put (Item => Float (Stop_Time - Start_Time), Fore => 3, Aft => 2, Exp => 0);
	 New_Line;
      end loop;
      New_Line;
   end Divide;


   -------------------
   -- Divide_Search --
   -------------------

   procedure Divide_Search (Max_Depth : in Natural) is
      Log           : File_Type;
      Letter        : array (0 .. 7) of Character := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' );
      Letter_Pieces : array (1 .. 11) of Character := ( ' ', 'p', 'p', 'n', 'n', 'b', 'b', 'r', 'r', 'q', 'q');
   begin
      if Max_Depth > 0 then
	 Generate_Moves;
	 for I in Ply_Type'Range loop
	    exit when not Is_Valid (Moves_List (Ply, I));
	    if Play (Moves_List (Ply, I)) then
	       if Max_Depth = 1 then
		  Nodes := Nodes + 1;
		  Open (Log, Append_File, "perft2.log");
		  Put (Log, Ply - 1, 0);
		  Put (Log, ". ");
		  for J in Ply_Type'First .. Ply - 1 loop
		     Put (Log, Letter (File (History_Moves (J).From)));
		     Put (Log, Rank (History_Moves (J).From), 0);
		     Put (Log, Letter (File (History_Moves (J).To)));
		     Put (Log, Rank (History_Moves (J).To), 0);
		     Put (Log, " ");
		  end loop;
		  Close (Log);
	       end if;
	       Divide_Search (Max_Depth => Max_Depth - 1);
	       Undo;
	    end if;
	 end loop;
      end if;
   end Divide_Search;


   -------------------------------
   -- Print_Principal_Variation --
   -------------------------------

   procedure Print_Principal_Variation (Score : in Integer) is
      Move : Move_Type;
   begin
      PV_Engine_Thinking_Time := Get_Thinked_Time;
      PV_Score := Float (Score) / 100.0;

      if Ouptut_Mode = Standard then
	 -- Print
	 -- 1) Search depth
	 -- 2) Thinking Time
	 -- 4) Nodes
	 -- 5) Score
	 -- 6) Principal Variation
	 Put (Search_Depth - 1, 0);
	 Put (" ");

	 Put (Item => Float (PV_Engine_Thinking_Time), Fore => 3, Aft => 2, Exp => 0);
	 Put (" ");
	 Put (Nodes);
	 Put ("  ");
	 Put (Item => Float (PV_Score), Fore => 5, Aft => 2, Exp => 0);
	 Put (" ");
	 for I in 1 .. Principal_Variation_Depth (Ply) - 1 loop
	    Move := Principal_Variation (1, I);
	    Echo (Move);
	    Put (" ");
	 end loop;
	 New_Line;
	 Flush;

      elsif Ouptut_Mode = Xboard then
	 -- Print
	 -- 1) Search depth
	 -- 2) Unformatted Score
	 -- 3) Unformatted Time
	 -- 4) Nodes
	 -- 5) Principal Variation
	 Put (Search_Depth - 1, 0);
	 Put (" ");
	 Put (Score, 0);
	 Put (" ");
	 Put (Integer ((PV_Engine_Thinking_Time * 100)));
	 Put (" ");
	 Put (Nodes);
	 Put ("  ");
	 for I in 1 .. Principal_Variation_Depth (Ply) - 1 loop
	    Move := Principal_Variation (1, I);
	    Echo (Move);
	    Put (" ");
	 end loop;
	 New_Line;
	 Flush;
      end if;

      Flush;

   end Print_Principal_Variation;


end ACSearch;
