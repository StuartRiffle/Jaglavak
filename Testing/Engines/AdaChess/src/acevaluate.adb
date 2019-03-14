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

-- pragma Optimize (Time);

package body ACEvaluate is


   -----------------------------------
   -- Draw_By_Insufficient_Material --
   -----------------------------------

   function Draw_By_Insufficient_Material return Boolean is
      White_Has_Insufficient_Material : Boolean := False;
      Score                     : Natural := 0;
   begin
      Score := 0;
      for I in White_Pieces'Range loop
	 exit when White_Pieces (I) = No_Piece;
	 case White_Pieces (I).Piece is
	    when White_Knight | White_Bishop =>
	       Score := Score + 1;
	    when White_Queen | White_Rook | White_Pawn =>
	       Score := Score + 10; -- a lot of score
	    when others => null;
	 end case;
	 exit when Score > 2;
      end loop;
      -- truly, white could mate even with a king and a knight,
      -- but in real life cannot do it so consider as a draw
      if Score <= 2 then -- white cant' mate
	 White_Has_Insufficient_Material := True;
      end if;

      Score := 0;
      for I in Black_Pieces'Range loop
	 exit when Black_Pieces (I) = No_Piece;
	 case Black_Pieces (I).Piece is
	    when Black_Knight | Black_Bishop =>
	       Score := Score + 1;
	    when Black_Queen | Black_Rook | Black_Pawn =>
	       Score := Score + 10; -- a lot of score
	    when others => null;
	 end case;
	 exit when Score > 2;
      end loop;

      if Score < 2 and then White_Has_Insufficient_Material then
	 return True; -- ya, it's a draw by insufficient material!
      end if;

      return False;

   end Draw_By_Insufficient_Material;


   --------------
   -- Evaluate --
   --------------

   function Evaluate return Integer is
      White_Score : Integer := 0;
      Black_Score : Integer := 0;
   begin

      White_Pawn_Position := (others => -1);
      White_Pawn_Counter := 0;
      White_Pawn_Rank := (others => 0);

      Black_Pawn_Position := (others => -1);
      Black_Pawn_Counter := 0;
      Black_Pawn_Rank := (others => 9);

      White_Pawn_File := (others => False);
      Black_Pawn_File := (others => False);

      White_Rooks_Position := (others => -1);
      Black_Rooks_Position := (others => -1);
      White_Rooks_Counter := 0;
      Black_Rooks_Counter := 0;

      White_Knights_Position := (others => -1);
      Black_Knights_Position := (others => -1);
      White_Knights_Counter := 0;
      Black_Knights_Counter := 0;

      White_Bishops_Position := (others => -1);
      Black_Bishops_Position := (others => -1);
      White_Bishops_Counter := 0;
      Black_Bishops_Counter := 0;

      White_Bishops_Color := (others => False);
      Black_Bishops_Color := (others => False);

      White_Queen_Position := 0;
      Black_Queen_Position := 0;

      White_Weak_Board := (others => False);
      Black_Weak_Board := (others => False);

      Count_White_Pieces := 0;
      Count_Black_Pieces := 0;


      for I in White_Pieces'Range loop
	 exit when White_Pieces (I) = No_Piece;
	 Count_White_Pieces := Count_White_Pieces + 1;
	 White_Score := White_Score - Distance (White_King_Position, White_Pieces (I).Square);
	 case White_Pieces (I).Piece is
	    when White_Pawn => White_Score := White_Score + Pawn_Score + Pawn_Square_Value (White_Pieces (I).Square);
	       White_Pawn_Position (White_Pawn_Position'First + White_Pawn_Counter) := White_Pieces (I).Square;
	       White_Pawn_Counter := White_Pawn_Counter + 1;
	       if White_Pawn_Rank (White_Pawn_Rank'First + File (White_Pieces (I).Square) + 1) < Rank (White_Pieces (I).Square) then
		  White_Pawn_Rank (White_Pawn_Rank'First + File (White_Pieces (I).Square) + 1) := Rank (White_Pieces (I).Square);
	       end if;
	       White_Pawn_File (File (White_Pieces (I).Square) + 1) := True;
	    when White_Knight => White_Score := White_Score + Knight_Score + Knight_Square_Value (White_Pieces (I).Square);
	       White_Knights_Position (White_Knights_Position'First + White_Knights_Counter) := White_Pieces (I).Square;
	       White_Knights_Counter := White_Knights_Counter + 1;
	    when White_Bishop => White_Score := White_Score + Bishop_Score + Bishop_Square_Value (White_Pieces (I).Square);
	       White_Bishops_Position (White_Bishops_Position'First + White_Bishops_Counter) := White_Pieces (I).Square;
	       White_Bishops_Counter := White_Bishops_Counter + 1;
	       White_Bishops_Color (Square_Color (White_Pieces (I).Square)) := True;
	    when White_Rook => White_Score := White_Score + Rook_Score; -- + Rook_Square_Value (White_Pieces (I).Square);
	       White_Rooks_Position (White_Rooks_Position'First + White_Rooks_Counter) := White_Pieces (I).Square;
	       White_Rooks_Counter := White_Rooks_Counter + 1;
	    when White_Queen => White_Score := White_Score + Queen_Score; -- + Queen_Square_Value (White_Pieces (I).Square);
	       White_Queen_Position := White_Pieces (I).Square;
--  	       if History_Ply > 20 then
--  		  White_Score := White_Score + Queen_Square_Value (White_Pieces (I).Square);
--  	       end if;
	    when White_King => White_Score := White_Score + King_Score; -- + King_Square_Value (White_Pieces (I).Square);
	    when others => null; -- raise exception
	 end case;
      end loop;

      for I in Black_Pieces'Range loop
	 exit when Black_Pieces (I) = No_Piece;
	 Count_Black_Pieces := Count_Black_Pieces + 1;
	 Black_Score := Black_Score - Distance (Black_King_Position, Black_Pieces (I).Square);
	 case Black_Pieces (I).Piece is
	    when Black_Pawn => Black_Score := Black_Score + Pawn_Score + Pawn_Square_Value (Flip (Black_Pieces (I).Square));
	       Black_Pawn_Position (Black_Pawn_Position'First + Black_Pawn_Counter) := Black_Pieces (I).Square;
	       Black_Pawn_Counter := Black_Pawn_Counter + 1;
	       if Black_Pawn_Rank ( File (Black_Pieces (I).Square) + 1) > Rank (Black_Pieces (I).Square) then
		  Black_Pawn_Rank ( File (Black_Pieces (I).Square) + 1) := Rank (Black_Pieces (I).Square);
	       end if;
	       Black_Pawn_File (File (Black_Pieces (I).Square) + 1) := True;
	    when Black_Knight => Black_Score := Black_Score + Knight_Score + Knight_Square_Value (Flip (Black_Pieces (I).Square));
	       Black_Knights_Position (Black_Knights_Position'First + Black_Knights_Counter) := Black_Pieces (I).Square;
	       Black_Knights_Counter := Black_Knights_Counter + 1;
	    when Black_Bishop => Black_Score := Black_Score + Bishop_Score + Bishop_Square_Value (Flip (Black_Pieces (I).Square));
	       Black_Bishops_Position (Black_Bishops_Position'First + Black_Bishops_Counter) := Black_Pieces (I).Square;
	       Black_Bishops_Counter := Black_Bishops_Counter + 1;
	       Black_Bishops_Color (Square_Color (Black_Pieces (I).Square)) := True;
	    when Black_Rook => Black_Score := Black_Score + Rook_Score;-- + Rook_Square_Value (Flip (Black_Pieces (I).Square));
	       Black_Rooks_Position (Black_Rooks_Position'First + Black_Rooks_Counter) := Black_Pieces (I).Square;
	       Black_Rooks_Counter := Black_Rooks_Counter + 1;
	    when Black_Queen => Black_Score := Black_Score + Queen_Score; -- + Queen_Square_Value (Flip (Black_Pieces (I).Square));
	       Black_Queen_Position := Black_Pieces (I).Square;
--  	       if History_Ply > 20 then
--  		  Black_Score := Black_Score + Queen_Square_Value (Flip (Black_Pieces (I).Square));
--  	       end if;
	    when Black_King => Black_Score := Black_Score + King_Score; -- + King_Square_Value (Flip (Black_Pieces (I).Square));
	    when others => null; -- raise exception
	 end case;
      end loop;


      -- detect if this is a draw by insufficient material
      if White_Score - King_Score <= 1000  and then Black_Score - King_Score <= 1000 then
	 if Draw_By_Insufficient_Material then
	    return Draw;
	 end if;
      end if;


      -- now we have to decide if we are still in the opening
      -- or else on the middle game or in and endgame
      if History_Ply < 30 then
	 Game_Status := Opening;
	 if Is_In_Board (White_Queen_Position) then
	    White_Score := White_Score + Queen_Square_Value (White_Queen_Position);
	 end if;
	 if Is_In_Board (Black_Queen_Position) then
	    Black_Score := Black_Score + Queen_Square_Value (Flip (Black_Queen_Position));
	 end if;
      elsif White_Score - King_Score + Black_Score - King_Score < Endgame_Score_Limit then
	 Game_Status := End_Game;
      else
	 Game_Status := Middle_Game;
      end if;


      White_Score := White_Score + Evaluate_White_Pawn_Structure;
      Black_Score := Black_Score + Evaluate_Black_Pawn_Structure;

      White_Score := White_Score + Evaluate_White_King_Safety;
      Black_Score := Black_Score + Evaluate_Black_King_Safety;

--        if Ply <= First_Evaluation_Limit then
--  	 if Side_To_Move = White then
--  	    return White_Score - Black_Score;
--  	 else
--  	    return Black_Score - White_Score;
--  	 end if;
--        end if;

      if Game_Status = Opening then
	 White_Score := White_Score + Evaluate_White_Piece_Development;
	 Black_Score := Black_Score + Evaluate_Black_Piece_Development;
      end if;

      White_Score := White_Score + Evaluate_White_Rooks;
      Black_Score := Black_Score + Evaluate_Black_Rooks;

      White_Score := White_Score + Evaluate_White_Material_Advantage (White_Score - Black_Score);
      Black_Score := Black_Score + Evaluate_Black_Material_Advantage (Black_Score - White_Score);


--         if Ply <= Second_Evaluation_Limit then
--  	 if Side_To_Move = White then
--  	    return White_Score - Black_Score;
--  	 else
--  	    return Black_Score - White_Score;
--  	 end if;
--        end if;

      Populate_Weak_Square_Board;
      White_Score := White_Score + Evaluate_White_Piece_Positional_Game;
      Black_Score := Black_Score + Evaluate_Black_Piece_Positional_Game;


      White_Score := White_Score + Evaluate_White_Mobility;
      Black_Score := Black_Score + Evaluate_Black_Mobility;

--         if Ply <= Third_Evaluation_Limit then
--  	 if Side_To_Move = White then
--  	    return White_Score - Black_Score;
--  	 else
--  	    return Black_Score - White_Score;
--  	 end if;
--        end if;


      White_Score := White_Score + Evaluate_White_Unprotected_Pieces;
      Black_Score := Black_Score + Evaluate_Black_Unprotected_Pieces;

      if Side_To_Move = White then
	 return White_Score - Black_Score;
      else
	 return Black_Score - White_Score;
      end if;

   end Evaluate;


   -----------------------------------
   -- Evaluate_White_Pawn_Structure --
   -----------------------------------

   function Evaluate_White_Pawn_Structure return Integer is
      Score             : Integer := 0;
      Pawn_File         : Integer := 0;
      Pawn_Rank         : Integer := 0;
      Count_White_Pawns : Integer := 0;
      Isolated, Backwards : Boolean;
   begin

      for I in White_Pawn_Position'Range loop
	 exit when White_Pawn_Position (I) = -1;
	 Isolated := False;
	 Backwards := False;
	 Pawn_File := File (White_Pawn_Position (I)) + 1;

	 Pawn_Rank := White_Pawn_Rank (Pawn_File);
	 -- is a doubled pawn?
	 if White_Pawn_Rank (Pawn_File) > Rank (White_Pawn_Position (I)) then
	    Score := Score - Doubled_Pawn_Penalty;
	 end if;
	 -- is an isolated pawn?
	 if White_Pawn_Rank (Pawn_File - 1) = 0 and then White_Pawn_Rank (Pawn_File + 1) = 0 then
	    Score := Score - Isolated_Pawn_Penalty;
	    Isolated := True;
	    -- maybe is backward?
	 elsif (White_Pawn_Rank (Pawn_File - 1) = 0 or else White_Pawn_Rank (Pawn_File - 1) > White_Pawn_Rank (Pawn_File))
	   and then (White_Pawn_Rank (Pawn_File + 1) = 0 or else White_Pawn_Rank (Pawn_File + 1) > White_Pawn_Rank (Pawn_File)) then
	    Score := Score - Backward_Pawn_Penalty;
	    Backwards := True;
	    -- check to see if it is furthermore isolated
	    if Isolated then
	       Score := Score - 3;
	    end if;
	 end if;
	 -- is this pawn exposed
	 if Black_Pawn_File (Pawn_File) = False then
	    if Isolated then
	       Score := Score - 8;
	    end if;
	    if Backwards then
	       Score := Score - 4;
	    end if;
	 end if;

--  	    is blocked?
	 if Is_Black (ChessBoard (White_Pawn_Position (I) + North)) then
	    if ChessBoard (White_Pawn_Position (I) + North) = Black_Pawn then
	       Score := Score - Blocked_Pawn_Penalty;
	    else
	       Score := Score - Partially_Blocked_Pawn_Penalty;
	    end if;
	 end if;
	 -- is passed?
	 if (Black_Pawn_Rank (Pawn_File - 1) = 9 or else Black_Pawn_Rank (Pawn_File - 1) <= White_Pawn_Rank (Pawn_File))
	   and then (Black_Pawn_Rank (Pawn_File) = 9 or else Black_Pawn_Rank (Pawn_File) <= White_Pawn_Rank (Pawn_File))
	   and then (Black_Pawn_Rank (Pawn_File + 1) = 9 or else Black_Pawn_Rank (Pawn_File + 1) <= White_Pawn_Rank (Pawn_File))
	 then
	    Score := Score + Passed_Pawn_Bonus + Pawn_Rank * Passed_Pawn_Score;
	 end if;
      end loop;

      return Score;
   end Evaluate_White_Pawn_Structure;


   -----------------------------------
   -- Evaluate_Black_Pawn_Structure --
   -----------------------------------

   function Evaluate_Black_Pawn_Structure return Integer is
      Score             : Integer := 0;
      Pawn_File         : Integer := 0;
      Pawn_Rank         : Integer := 0;
      Count_Black_Pawns : Integer := 0;
      Isolated, Backwards : Boolean;
   begin
      for I in Black_Pawn_Position'Range loop
	 exit when Black_Pawn_Position (I) = -1;
	 Isolated := False;
	 Backwards := False;
	 Pawn_File := File (Black_Pawn_Position (I)) + 1;
	 Pawn_Rank := Black_Pawn_Rank (Pawn_File);

	 -- is a doubled pawn?
	 -- a pawn is doubled when there's another pawn on the
	 -- same file. There can be more than two pawns on the same file.
	 if Black_Pawn_Rank (Pawn_File) < Rank (Black_Pawn_Position (I)) then
	    Score := Score - Doubled_Pawn_Penalty;
	 end if;
	 -- is an isolated pawn?
	 if Black_Pawn_Rank (Pawn_File - 1) = 9 and then Black_Pawn_Rank (Pawn_File + 1) = 9 then
	    Score := Score - Isolated_Pawn_Penalty;
	    Isolated := True;
	    -- maybe is backward?
	 elsif (Black_Pawn_Rank (Pawn_File - 1) = 9 or else Black_Pawn_Rank (Pawn_File - 1) < Black_Pawn_Rank (Pawn_File))
	   and then (Black_Pawn_Rank (Pawn_File + 1) = 9 or else Black_Pawn_Rank (Pawn_File + 1) < Black_Pawn_Rank (Pawn_File)) then
	    Score := Score - Backward_Pawn_Penalty;
	    Backwards := True;
	    if Isolated then
	       Score := Score - 3;
	    end if;
	 end if;
	 -- is this pawn exposed
	  if White_Pawn_File (Pawn_File) = False then
	    if Isolated then
	       Score := Score - 4;
	    end if;
	    if Backwards then
	       Score := Score - 2;
	    end if;
	 end if;
	 -- is blocked?
	 if Is_White (ChessBoard (Black_Pawn_Position (I) + South)) then
	    if ChessBoard (Black_Pawn_Position (I) + South) = White_Pawn then
	       Score := Score - Blocked_Pawn_Penalty;
	    else
	       Score := Score - Partially_Blocked_Pawn_Penalty;
	    end if;
	 end if;
	 -- is passed?
	 if (White_Pawn_Rank (Pawn_File - 1) = 0 or else White_Pawn_Rank (Pawn_File - 1) >= Black_Pawn_Rank (Pawn_File))
	   and then (White_Pawn_Rank (Pawn_File) = 0 or else White_Pawn_Rank (Pawn_File) >= Black_Pawn_Rank (Pawn_File))
	   and then (White_Pawn_Rank (Pawn_File + 1) = 0 or else White_Pawn_Rank (Pawn_File + 1) >= Black_Pawn_Rank (Pawn_File))
	 then
	    Score := Score + Passed_Pawn_Bonus + (9 - Pawn_Rank) * Passed_Pawn_Score;
	 end if;
      end loop;

      return Score;

   end Evaluate_Black_Pawn_Structure;


   --------------------------------
   -- Evaluate_White_King_Safety --
   --------------------------------

   function Evaluate_White_King_Safety return Integer is
      Score       : Integer := 0;
      Has_Castled : Boolean := False;
   begin

--        for I in White_Pieces'Range loop
--  	 exit when White_Pieces (I) = No_Piece;
--  	 if Distance (White_Pieces (I).Square, White_King_Position) > 3 then
--  	    Score := Score - 1;
--  	 end if;
--        end loop;

      if Game_Status = End_Game then
	 return King_End_Game_Square_Value (White_King_Position);
      else
	 Score := Score + King_Square_Value (White_King_Position);
	 if History_Ply in 9 .. 30 then
	    for I in reverse History_Ply - 7 .. History_Ply - 1 loop
	       if History_Moves (I).Piece = White_King
		 and then History_Moves (I).From = E1
		 and then (History_Moves (I).To = G1 or else History_Moves (I).To = C1) then
		  Has_Castled := True;
		  if History_Moves (I).To = C1 then -- encourage long castle
		     Score := Score + 10;
		  end if;
	       end if;
	       exit when Has_Castled = True;
	    end loop;
	 end if;
      end if;


      if Has_Castled then
	 Score := Score + King_Has_Castled_Bonus;
	 -- if king has caslte, then check if castle is weak!
	 if Rank (White_King_Position) = 1 then
	    if File (White_King_Position) < 3 then
	       if White_King_Position = C1 then
		  Score := Score - King_Castle_Expose_Penalty;
	       end if;
	       if ChessBoard (A2) = White_Pawn and then ChessBoard (B2) = White_Pawn and then ChessBoard (C2) = White_Pawn then
		  Score := Score + King_Has_Castled_Protection_Bonus;
	       elsif ChessBoard (A3) = White_Pawn and then ChessBoard (B2) = White_Pawn and then ChessBoard (C2) = White_Pawn then
		  Score := Score + King_Has_Castled_Half_Bonus;
	       elsif ChessBoard (A2) = White_Pawn and then ChessBoard (B3) = White_Pawn and then ChessBoard (C2) = White_Pawn then
		  Score := Score + King_Has_Castled_With_Hole_Bonus;
	       elsif ChessBoard (A2) = White_Pawn and then ChessBoard (B2) = White_Pawn and then ChessBoard (C3) = White_Pawn then
		  Score := Score + King_Has_Castled_Corrupted_Bonus;
	       else
		  if Is_Empty (A2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (B2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (C2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
	       end if;
	       if Chessboard (C3) = White_Knight then
		  Score := Score + King_Protected_By_Knight_Bonus;
		   if Is_White (ChessBoard (D4)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       elsif Is_White (Chessboard (C3)) then
		  Score := Score + King_Protected_By_Friendly_Bonus;
		   if Is_White (ChessBoard (D4)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       end if;

	    elsif File (White_King_Position) > 4 then
	       --  	   Put_Line ("White king has castled to the kingnside");
	       if ChessBoard (H2) = White_Pawn and then ChessBoard (G2) = White_Pawn and then ChessBoard (F2) = White_Pawn then
		  Score := Score + King_Has_Castled_Protection_Bonus;
	       elsif ChessBoard (H3) = White_Pawn and then ChessBoard (G2) = White_Pawn and then ChessBoard (F2) = White_Pawn then
		  Score := Score + King_Has_Castled_Half_Bonus;
	       elsif ChessBoard (H2) = White_Pawn and then ChessBoard (G3) = White_Pawn and then ChessBoard (F2) = White_Pawn then
		  if ChessBoard (G2) = White_Bishop then -- fianchetto
		     Score := Score + King_Has_Castled_With_Fianchetto_Bonus;
		  else
		     Score := Score + King_Has_Castled_With_Hole_Bonus;
		  end if;
	       elsif ChessBoard (H2) = White_Pawn and then ChessBoard (G2) = White_Pawn and then ChessBoard (F3) = White_Pawn then
		  Score := Score + King_Has_Castled_Corrupted_Bonus;
	       else
		  if Is_Empty (H2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (G2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (F2) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
	       end if;
	       if Chessboard (F3) = White_Knight then
		  Score := Score + King_Protected_By_Knight_Bonus;
		  if Is_White (ChessBoard (E4)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       elsif Is_White (Chessboard (C3)) then
		  Score := Score + King_Protected_By_Friendly_Bonus;
		  if Is_White (ChessBoard (E4)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       end if;
	    end if;

	 end if;
      else
	 -- still can castle?
	 if History_Ply < 30 then
	    if Castle.White_Kingside or else Castle.White_Queenside then
	       Score := Score - King_Has_Moved_Before_Castling_Penalty / 2;
	    else
	       Score := Score - King_Has_Moved_Before_Castling_Penalty;
	    end if;
	 end if;

      end if;

      return Score;

   end Evaluate_White_King_Safety;


   --------------------------------
   -- Evaluate_Black_King_Safety --
   --------------------------------

   function Evaluate_Black_King_Safety return Integer is
      Score       : Integer := 0;
      Has_Castled : Boolean := False;
   begin

--        for I in Black_Pieces'Range loop
--  	 exit when Black_Pieces (I) = No_Piece;
--  	 if Distance (Black_Pieces (I).Square, Black_King_Position) > 3 then
--  	    Score := Score - 1;
--  	 end if;
--        end loop;

      if Game_Status = End_Game then
	return King_End_Game_Square_Value (Black_King_Position); -- table is simmetrical as white king
      else
	 Score := Score + King_Square_Value (Flip (Black_King_Position));
	 if History_Ply in 9 .. 30 then
	    for I in reverse History_Ply - 7 .. History_Ply - 1 loop
	       if History_Moves (I).Piece = Black_King
		 and then History_Moves (I).From = E8
		 and then (History_Moves (I).To = G8 or else History_Moves (I).To = C8) then
		  Has_Castled := True;
		  if History_Moves (I).To = C8 then -- encourage long castle
		     Score := Score + 10;
		  end if;
	       end if;
	       exit when Has_Castled = True;
	    end loop;
	 end if;
      end if;

      if Has_Castled then
	 Score := Score + King_Has_Castled_Bonus;
	 if Rank (Black_King_Position) = 8 then
	    if File (Black_King_Position) < 3 then
	       if Black_King_Position = C8 then
		  Score := Score - King_Castle_Expose_Penalty;
	       end if;
	       if ChessBoard (A7) = Black_Pawn and then ChessBoard (B7) = Black_Pawn and then ChessBoard (C7) = Black_Pawn then
		  Score := Score + King_Has_Castled_Protection_Bonus;
	       elsif ChessBoard (A6) = Black_Pawn and then ChessBoard (B7) = Black_Pawn and then ChessBoard (C7) = Black_Pawn then
		  Score := Score + King_Has_Castled_Half_Bonus;
	       elsif ChessBoard (A7) = Black_Pawn and then ChessBoard (B6) = Black_Pawn and then ChessBoard (C7) = Black_Pawn then
		  Score := Score + King_Has_Castled_With_Hole_Bonus;
	       elsif ChessBoard (A7) = Black_Pawn and then ChessBoard (B7) = Black_Pawn and then ChessBoard (C6) = Black_Pawn then
		  Score := Score + King_Has_Castled_Corrupted_Bonus;
	       else
		  if Is_Empty (A6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (B6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (C6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
	       end if;
	       if Chessboard (C6) = Black_Knight then
		  Score := Score + King_Protected_By_Knight_Bonus;
		   if Is_Black (ChessBoard (D5)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       elsif Is_Black (Chessboard (C6)) then
		  Score := Score + King_Protected_By_Friendly_Bonus;
		   if Is_Black (ChessBoard (D5)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       end if;
	    elsif File (Black_King_Position) > 4 then
	       if ChessBoard (H7) = Black_Pawn and then ChessBoard (G7) = Black_Pawn and then ChessBoard (F7) = Black_Pawn then
		  Score := Score + King_Has_Castled_Protection_Bonus;
	       elsif ChessBoard (H6) = Black_Pawn and then ChessBoard (G7) = Black_Pawn and then ChessBoard (F7) = Black_Pawn then
		  Score := Score + King_Has_Castled_Half_Bonus;
	       elsif ChessBoard (H7) = Black_Pawn and then ChessBoard (G6) = Black_Pawn and then ChessBoard (F7) = Black_Pawn then
		  if ChessBoard (G7) = Black_Bishop then -- fianchetto
		     Score := Score + King_Has_Castled_With_Fianchetto_Bonus;
		  else
		     Score := Score + King_Has_Castled_With_Hole_Bonus;
		  end if;
	       elsif ChessBoard (H7) = Black_Pawn and then ChessBoard (G7) = Black_Pawn and then ChessBoard (F6) = Black_Pawn then
		  Score := Score + King_Has_Castled_Corrupted_Bonus;
	       else
		  if Is_Empty (H6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (G6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
		  if Is_Empty (F6) then
		     Score := Score - King_Castle_Without_A_Pawn_Penalty;
		  end if;
	       end if;
	       if Chessboard (F6) = Black_Knight then
		  Score := Score + King_Protected_By_Knight_Bonus;
		  if Is_Black (ChessBoard (E5)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       elsif Is_Black (Chessboard (F6)) then
		  Score := Score + King_Protected_By_Friendly_Bonus;
		  if Is_Black (ChessBoard (E5)) then
		     Score := Score + King_Protected_By_Friendly_Bonus;
		  end if;
	       end if;
	    end if;
	 end if;
      else
	 -- still can castle?
	 if History_Ply < 30 then
	    if Castle.Black_Kingside or else Castle.Black_Queenside then
	       Score := Score - King_Has_Moved_Before_Castling_Penalty / 2;
	    else
	       Score := Score - King_Has_Moved_Before_Castling_Penalty;
	    end if;
	 end if;

      end if;

      return Score;

   end Evaluate_Black_King_Safety;


   --------------------------
   -- Evaluate_White_Rooks --
   --------------------------

   function Evaluate_White_Rooks return Integer is
      Score : Integer := 0;
      Rook_File : Integer := 0;
      Rook_Rank : Integer := 0;
      Other_Rook_File      : Integer := 0;
      Other_Rook_Rank      : Integer := 0;
      Black_King_File      : Integer;
   begin
      for I in White_Rooks_Position'Range loop
	 exit when White_Rooks_Position (I) < 0;
	 Rook_File := File (White_Rooks_Position (I)) + 1;
	 Rook_Rank := Rank (White_Rooks_Position (I));
	 if not White_Pawn_File (Rook_File) and then not Black_Pawn_File (Rook_File) then
	    Score := Score + Rook_On_Open_File_Bonus;
	    if Game_Status = End_Game then
	       -- limit opponent king movement
	       Black_King_File := File (Black_King_Position) + 1;
	       if Black_King_File < 4 and then Rook_File = Black_King_File + 1 then
		  Score := Score + 10;
	       elsif Black_King_File > 4 and then Rook_File = Black_King_File - 1 then
		  score := Score + 10;
	       end if;
	    end if;
	 elsif not White_Pawn_File (Rook_File) and then Black_Pawn_File (Rook_File) then
	    Score := Score + Rook_On_Semi_Open_File_Bonus;
	 end if;
	 if Rook_Rank = 7 then
	    Score := Score + Rook_On_Seventh_Rank_Bonus;
	 end if;
	 -- look for rooks on same rank/file
	 for J in White_Rooks_Position'Range loop
	    exit when White_Rooks_Position (J) < 0;
	    Other_Rook_File := File (White_Rooks_Position (J));
	    Other_Rook_Rank := Rank (White_Rooks_Position (J));
	    if Rook_File = Other_Rook_File and then Rook_Rank /= Other_Rook_Rank then
	       Score := Score + Rooks_On_Same_File_Bonus;
	    elsif Rook_File /= Other_Rook_File and then Rook_Rank = Other_Rook_Rank then
	       Score := Score + Rooks_On_Same_Rank_Bonus;
	    end if;
	 end loop;
      end loop;
      return Score;
   end Evaluate_White_Rooks;


   --------------------------
   -- Evaluate_Black_Rooks --
   --------------------------

   function Evaluate_Black_Rooks return Integer is
      Score          : Integer := 0;
      Rook_File      : Integer := 0;
      Rook_Rank      : Integer := 0;
      Other_Rook_File      : Integer := 0;
      Other_Rook_Rank      : Integer := 0;
      White_King_File      : Integer;
   begin
      for I in Black_Rooks_Position'Range loop
	 exit when Black_Rooks_Position (I) < 0;
	 Rook_File := File (Black_Rooks_Position (I)) + 1;
	 Rook_Rank := Rank (Black_Rooks_Position (I));
	 -- find rooks on open/semiopen file
	 if not Black_Pawn_File (Rook_File) and then not White_Pawn_File (Rook_File) then
	    Score := Score + Rook_On_Open_File_Bonus;
	    if Game_Status = End_Game then
	       -- limit opponent king movement
	       White_King_File := File (White_King_Position) + 1;
	       if White_King_File < 4 and then Rook_File = White_King_File + 1 then
		  Score := Score + 10;
	       elsif White_King_File > 4 and then Rook_File = White_King_File - 1 then
		  Score := SCore + 10;
	       end if;
	    end if;
	 elsif not Black_Pawn_File (Rook_File) and then White_Pawn_File (Rook_File) then
	    Score := Score + Rook_On_Semi_Open_File_Bonus;
	 end if;
	 -- give a bonus for rook on sevent
	 if Rook_Rank = 2 then
	    Score := Score + Rook_On_Seventh_Rank_Bonus;
	 end if;
	 -- look for rooks on same rank/file
	 for J in Black_Rooks_Position'Range loop
	    exit when Black_Rooks_Position (J) < 0;
	    Other_Rook_File := File (Black_Rooks_Position (J));
	    Other_Rook_Rank := Rank (Black_Rooks_Position (J));
	    if Rook_File = Other_Rook_File and then Rook_Rank /= Other_Rook_Rank then
	       Score := Score + Rooks_On_Same_File_Bonus;
	    elsif Rook_File /= Other_Rook_File and then Rook_Rank = Other_Rook_Rank then
	       Score := Score + Rooks_On_Same_Rank_Bonus;
	    end if;
	 end loop;
      end loop;
      return Score;
   end Evaluate_Black_Rooks;


   --------------------------------------
   -- Evaluate_White_Piece_Development --
   --------------------------------------

   function Evaluate_White_Piece_Development return Integer is
      Score              : Integer := 0;
      Development_Status : Integer := 0;
   begin
      -- discourage queen moves on opening
      if History_Ply in 7 .. 14 then
	 for I in reverse 6 .. History_Ply - 1 loop
	    if History_Moves (I).Piece = White_Queen then
	       Score := Score - Queen_Moves_On_First_Moves_Penalty;
	    end if;
	    exit when Score /= 0;
	 end loop;
      end if;

      -- a good develop envolves at least 3 pieces from knights and bishops
      -- ad both pawns from d2 and e4
      for I in 1 .. History_Ply - 1 loop
	 case History_Moves (I).Piece is
	    when White_Pawn =>
	       if History_Moves (I).From = D2 then
		  if History_Moves (I).To = D3 then
		     Development_Status := Development_Status + 4;
		  else -- moves d2d4
		     Development_Status := Development_Status + 8;
		  end if;
	       elsif History_Moves (I).From = E2 then
		  if History_Moves (I).To = E3 then
		     Development_Status := Development_Status + 4;
		  else
		     Development_Status := Development_Status + 8;
		  end if;
	       end if;
	    when White_Knight => Development_Status := Development_Status + 10;
	    when White_Bishop => Development_Status := Development_Status + 8;
	    when others => null;
	 end case;
      end loop;

      -- special case of fianchetto is treated here:
      if ChessBoard (H2) = White_Pawn and then
	ChessBoard (G3) = White_Pawn and then
	ChessBoard (F2) = White_Pawn and then
	ChessBoard (G2) = White_Bishop and then
	ChessBoard (F3) = White_Knight and then
	ChessBoard (G1) = White_King then
	 Development_Status := Development_Status + 12;
      end if;

      -- not a good opening? return a bad value!
      if Development_Status < 50 then
	 return -100 + Score;
      end if;

      return Development_Status + Score;

   end Evaluate_White_Piece_Development;


   --------------------------------------
   -- Evaluate_Black_Piece_Development --
   --------------------------------------

   function Evaluate_Black_Piece_Development return Integer is
      Score : Integer := 0;
      Development_Status : Integer := 0;
   begin
      -- discourage queen moves on opening
      if History_Ply in 7 .. 14 then
	 for I in reverse 6 .. History_Ply - 1 loop
	    if History_Moves (I).Piece = Black_Queen then
	       Score := Score - Queen_Moves_On_First_Moves_Penalty;
	    end if;
	    exit when Score /= 0;
	 end loop;
      end if;

      -- a good develop envolves at least 3 pieces from knights and bishops
      -- ad both pawns from d2 and e4
      for I in 1 .. History_Ply - 1 loop
	 case History_Moves (I).Piece is
	    when Black_Pawn =>
	       if History_Moves (I).From = D7 then
		  if History_Moves (I).To = D6 then
		     Development_Status := Development_Status + 4;
		  else -- moves d2d4
		     Development_Status := Development_Status + 8;
		  end if;
	       elsif History_Moves (I).From = E7 then
		  if History_Moves (I).To = E6 then
		     Development_Status := Development_Status + 4;
		  else
		     Development_Status := Development_Status + 8;
		  end if;
	       end if;
	    when Black_Knight => Development_Status := Development_Status + 10;
	    when Black_Bishop => Development_Status := Development_Status + 8;
	    when others => null;
	 end case;
      end loop;

      -- special case of fianchetto is treated here:
      if ChessBoard (H7) = Black_Pawn and then
	ChessBoard (G6) = Black_Pawn and then
	ChessBoard (F7) = Black_Pawn and then
	ChessBoard (G7) = Black_Bishop and then
	ChessBoard (F7) = Black_Knight and then
	ChessBoard (G8) = Black_King then
	 Development_Status := Development_Status + 12;
      end if;

      if Development_Status < 50 then
	 return -100 + Score;
      end if;

      return Development_Status + Score;

   end Evaluate_Black_Piece_Development;


   ------------------------------------------
   -- Evaluate_White_Piece_Positional_Game --
   ------------------------------------------

   function Evaluate_White_Piece_Positional_Game return Integer is
      Score : Integer := 0;
      Square_Score : Integer := 0;
   begin
      for I in White_Pieces'Range loop
	 exit when White_Pieces (I) = No_Piece;
	 if Black_Weak_Board (White_Pieces (I).Square) then
	    case White_Pieces (I).Piece is
	       when White_Pawn => Score := Score + Occupying_Weak_Square_Bonus;
	       when White_Knight => Score := Score + Occupying_Weak_Square_Bonus;
	       when White_Bishop => Score := Score + Occupying_Weak_Square_Bonus;
	       when White_Rook => Score := Score + Occupying_Weak_Square_Bonus;
	       when others => null;
	    end case;
--  	    case White_Pieces (I).Square is
--  	       when E3 | E4 | D3 | D4 =>
--  		  Score := Score + 20;
--  	       when others => null;
--  	    end case;
	 end if;
      end loop;

--        for I in Black_Weak_Square'Range loop
--  	 exit when Black_Weak_Square (I) < 0;
--  	 case ChessBoard (Black_Weak_Square (I)) is
--  	    when White_Knight =>  Score := Score + Occupying_Weak_Square_Bonus;
--  	    when White_Bishop => Score := Score + Occupying_Weak_Square_Bonus;
--  	    when White_Rook => Score := Score + Occupying_Weak_Square_Bonus;
--  	    when others => null;
--  	 end case;
--        end loop;

      -- give penalty for blocking d/e pawns
      if ChessBoard (D2) = White_Pawn and then Is_White (ChessBoard (D3)) then
	 Score := Score - Pawn_Blocked_On_Center_Penalty;
      end if;
      if ChessBoard (E2) = White_Pawn and then Is_White (ChessBoard (E4)) then
	 Score := Score - Pawn_Blocked_On_Center_Penalty;
      end if;


      -- look for trapped bishop on 7th rank
      if (Chessboard (A7) = White_Bishop and then ChessBoard (B6) = Black_Pawn)
	or else (Chessboard (B8) = White_Bishop and then ChessBoard (C7) = Black_Pawn) then
	 Score := Score - Trapped_Bishop_Penalty;
      end if;
      if (Chessboard (H7) = White_Bishop and then ChessBoard (G6) = Black_Pawn)
	or else (Chessboard (G8) = White_Bishop and then ChessBoard (F7) = Black_Pawn) then
	 Score := Score - Trapped_Bishop_Penalty;
      end if;
      -- trapped bishop on 6th rank
      if ChessBoard (A6) = White_Bishop and then Chessboard (B5) = Black_Pawn then
	 Score := Score - Trapped_Bishop_Half_Penalty;
      end if;
      if Chessboard (H6) = White_Bishop and then ChessBoard (G5) = Black_Pawn then
	 Score := Score - Trapped_Bishop_Half_Penalty;
      end if;
      -- blocked bishop
      if ChessBoard (D2) = White_Pawn and then not Is_Empty (D3) and then Chessboard (C1) = White_Bishop then
	 Score := Score - Blocked_Bishop_Penalty;
      end if;
      if Chessboard (E2) = White_Pawn and then not Is_Empty (E3) and then Chessboard (F1) = White_Bishop then
	 Score := Score - Blocked_Bishop_Penalty;
      end if;

      -- give an extra score for bishop pairs
      if White_Bishops_Color (White) = True and then White_Bishops_Color (Black) = True then
	 Score := Score + Bishop_Pairs_Bonus;
      end if;

      -- blocked rook
      if (Chessboard (C1) = White_King or else Chessboard (B1) = White_King)
	and then (Chessboard (A1) = White_Rook or else Chessboard (A2) = White_Rook or else Chessboard (B1) = White_Rook)
      then
	 Score := Score - Blocked_Rook_Penalty;
      end if;
      if (Chessboard (F1) = White_King or else Chessboard (G1) = White_King)
	and then (Chessboard (H1) = White_Rook or else Chessboard (H2) = White_Rook or else Chessboard (G1) = White_Rook)
      then
	 Score := Score - Blocked_Rook_Penalty;
      end if;

      -- pawn blocked on center
      if Chessboard (D4) = Black_Pawn then
	 if Chessboard (D3) = White_Pawn then
	    Score := Score - Pawn_Blocked_On_Center_Penalty;
	 else
	    Score := Score - Pawn_Free_To_Kill_Penalty;
	 end if;
      end if;
      if  Chessboard (E4) = Black_Pawn then
	 if Chessboard (E3) = White_Pawn then
	    Score := Score - Pawn_Blocked_On_Center_Penalty;
	 else
	    Score := Score - Pawn_Free_To_Kill_Penalty;
	 end if;
      end if;

      return Score;
   end Evaluate_White_Piece_Positional_Game;


   ------------------------------------------
   -- Evaluate_Black_Piece_Positional_Game --
   ------------------------------------------

   function Evaluate_Black_Piece_Positional_Game return Integer is
      Score : Integer := 0;
      Square_Score : Integer := 0;
   begin
      for I in Black_Pieces'Range loop
	 exit when Black_Pieces (I) = No_Piece;
	 if White_Weak_Board (Black_Pieces (I).Square) then
	    case Black_Pieces (I).Piece is
	       when Black_Pawn => Score := Score + Occupying_Weak_Square_Bonus;
	       when Black_Knight => Score := Score + Occupying_Weak_Square_Bonus;
	       when Black_Bishop => Score := Score + Occupying_Weak_Square_Bonus;
	       when Black_Rook => Score := Score + Occupying_Weak_Square_Bonus;
	       when others => null;
	    end case;
--  	    case Black_Pieces (I).Square is
--  	       when E3 | E4 | D3 | D4 =>
--  		  Score := Score + 20;
--  	       when others => null;
--  	    end case;
	 end if;
      end loop;
--        for I in White_Weak_Square'Range loop
--  	 exit when White_Weak_Square (I) < 0;
--  	 case ChessBoard (White_Weak_Square (I)) is
--  	    when Black_Knight =>  Score := Score + Occupying_Weak_Square_Bonus;
--  	    when Black_Bishop => Score := Score + Occupying_Weak_Square_Bonus;
--  	    when Black_Rook => Score := Score + Occupying_Weak_Square_Bonus;
--  	    when others => null;
--  	 end case;
--        end loop;

       -- look for trapped bishop on 2nd rank
      if (Chessboard (A2) = Black_Bishop and then ChessBoard (B3) = White_Pawn)
	or else (Chessboard (B1) = Black_Bishop and then ChessBoard (C2) = White_Pawn) then
	 Score := Score - Trapped_Bishop_Penalty;
      end if;
      if (Chessboard (H2) = Black_Bishop and then ChessBoard (G3) = White_Pawn)
	or else (Chessboard (G1) = Black_Bishop and then ChessBoard (F2) = White_Pawn) then
	 Score := Score - Trapped_Bishop_Penalty;
      end if;
       -- trapped bishop on 3rd rank
      if ChessBoard (A3) = White_Bishop and then Chessboard (B4) = Black_Pawn then
	 Score := Score - Trapped_Bishop_Half_Penalty;
      end if;
       if Chessboard (H3) = White_Bishop and then ChessBoard (G4) = Black_Pawn then
	 Score := Score - Trapped_Bishop_Half_Penalty;
      end if;
      -- blocked bishop
      if Chessboard (D7) = Black_Pawn and then not Is_Empty (D6) and then Chessboard (C8) = Black_Bishop then
	 Score := Score - Blocked_Bishop_Penalty;
      end if;
      if Chessboard (E7) = Black_Pawn and then not Is_Empty (E6) and then Chessboard (F8) = Black_Bishop then
	 Score := Score - Blocked_Bishop_Penalty;
      end if;

       -- give an extra score for bishop pairs
      if Black_Bishops_Color (White) = True and then Black_Bishops_Color (Black) = True then
	 Score := Score + Bishop_Pairs_Bonus;
      end if;

      -- give penalty for blocking d/e pawns
      if ChessBoard (D7) = White_Pawn and then Is_White (ChessBoard (D6)) then
	 Score := Score - Pawn_Blocked_On_Center_Penalty;
      end if;
      if ChessBoard (E7) = White_Pawn and then Is_White (ChessBoard (E6)) then
	 Score := Score - Pawn_Blocked_On_Center_Penalty;
      end if;


      -- blocked rook
      if (Chessboard (C8) = Black_King or else Chessboard (B8) = Black_King)
	and then (Chessboard (A8) = Black_Rook or else Chessboard (A7) = Black_Rook or else Chessboard (B8) = Black_Rook)
      then
	 Score := Score - Blocked_Rook_Penalty;
      end if;
      if (Chessboard (F8) = Black_King or else Chessboard (G8) = Black_King)
	and then (Chessboard (H8) = Black_Rook or else Chessboard (H7) = Black_Rook or else Chessboard (G8) = Black_Rook)
      then
	 Score := Score - Blocked_Rook_Penalty;
      end if;

      -- opponent pawn controlling center
      if ChessBoard (D5) = White_Pawn then
	 if Chessboard (D6) = Black_Pawn then
	    Score := Score - Pawn_Blocked_On_Center_Penalty;
	 else
	    Score := Score - Pawn_Free_To_Kill_Penalty;
	 end if;
      end if;
      if Chessboard (E5) = White_Pawn then
	 if Chessboard (E6) = Black_Pawn then
	    Score := Score - Pawn_Blocked_On_Center_Penalty;
	 else
	    Score := Score - Pawn_Free_To_Kill_Penalty;
	 end if;
      end if;

      return Score;
   end Evaluate_Black_Piece_Positional_Game;


   -----------------------------
   -- Evaluate_White_Mobility --
   -----------------------------

   -- Detect:
   -- 1) Pinning pieces and (potential) discovery attacks and discovery checks
   -- 2) Unprotected piece
   -- 3) Forks
   function Evaluate_White_Mobility return Integer is
      Score : Integer := 0;
      Square  : Integer;
      Target, X_Ray_Target : Integer;
      Pins_Occured         : Boolean;
   begin
      -- 1) Pinning pieces. Looks if Bishop and Rook are pinning
      --    some black pieces
      Pins_Occured := False;
      for I in White_Bishops_Position'Range loop
	 exit when White_Bishops_Position (I) = -1;
	 Square := White_Bishops_Position (I);
	 for Offset in Bishop_Offsets'Range loop
	    exit when Pins_Occured; -- once we found a pin, is very rare that a piece pins twice in the same position!
	    Target := Square + Bishop_Offsets(Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Bishop_Offsets(Offset);
	    end loop;
	    -- here's we can detect if bishop is pinning!
	    if Is_Black (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Bishop_Offsets (Offset);
		  if Is_Black (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
	 end loop;
      end loop;
      Pins_Occured := False;
      for I in White_Rooks_Position'Range loop
	 exit when White_Rooks_Position (I) = -1;
	 Square := White_Rooks_Position (I);
	 for Offset in Rook_Offsets'Range loop
	    exit when Pins_Occured;
	    Target := Square + Rook_Offsets(Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Rook_Offsets(Offset);
	    end loop;
	    -- here's we can detect if rook is pinning!
	    if Is_Black (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Rook_Offsets(Offset);
		  if Is_Black (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
	 end loop;
      end loop;
      if Is_In_Board(White_Queen_Position) then
	 Pins_Occured := False;
	 for Offset in Queen_Offsets'Range loop
	    exit when Pins_Occured;
	    Target := White_Queen_Position + Queen_Offsets (Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Queen_Offsets (Offset);
	    end loop;
	    -- here's we can detect if rook is pinning!
	    if Is_Black (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Queen_Offsets (Offset);
		  if Is_Black (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
	 end loop;
      end if;
      return Score;
   end Evaluate_White_Mobility;


   -----------------------------
   -- Evaluate_Black_Mobility --
   -----------------------------

   function Evaluate_Black_Mobility return Integer is
      Score                : Integer := 0;
      Square               : Integer;
      Target, X_Ray_Target : Integer;
      Pins_Occured         : Boolean;
   begin
      -- 1) Pinning pieces. Looks if Bishop and Rook are pinning
      --    some black pieces
      Pins_Occured := False;
      for I in Black_Bishops_Position'Range loop
	 exit when Black_Bishops_Position (I) = -1;
	 Square := Black_Bishops_Position (I);
	 for Offset in Bishop_Offsets'Range loop
	    exit when Pins_Occured;
	    Target := Square + Bishop_Offsets(Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Bishop_Offsets(Offset);
	    end loop;
	    -- here's we can detect if bishop is pinning!
	    if Is_White (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Bishop_Offsets(Offset);
		  if Is_White (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
	 end loop;
      end loop;
      Pins_Occured := False;
      for I in Black_Rooks_Position'Range loop
	 exit when Black_Rooks_Position (I) = -1;
	 Square := Black_Rooks_Position (I);
	 for Offset in Rook_Offsets'Range loop
	    exit when Pins_Occured;
	    Target := Square + Rook_Offsets(Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Rook_Offsets(Offset);
	    end loop;
	    -- here's we can detect if rook is pinning!
	    if Is_White (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Rook_Offsets(Offset);
		  if Is_White (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
      end loop;
      end loop;
      -- find pins for black queen
      if Is_In_Board (Black_Queen_Position) then
	 Pins_Occured := False;
	 for Offset in Queen_Offsets'Range loop
	    exit when Pins_Occured;
	    Target := Black_Queen_Position + Queen_Offsets (Offset);
	    while Is_In_Board (Target) loop
	       exit when not Is_Empty (Target);
	       Target := Target + Queen_Offsets (Offset);
	    end loop;
	    -- here's we can detect if rook is pinning!
	    if Is_White (Chessboard (Target)) then
	       X_Ray_Target := Target;
	       while Is_In_Board (X_Ray_Target) loop
		  X_Ray_Target := X_Ray_Target + Queen_Offsets (Offset);
		  if Is_White (ChessBoard (X_Ray_Target))
		    and then Pinning_Piece_Table (ChessBoard (Target), ChessBoard (X_Ray_Target)) then
		     Score := Score + Pinning_Piece_Bonus;
		     Pins_Occured := True;
		  end if;
		  exit when not Is_Empty (X_Ray_Target);
	       end loop;
	    end if;
	 end loop;
      end if;
      return Score;
   end Evaluate_Black_Mobility;


   ---------------------------------------
   -- Evaluate_White_Unprotected_Pieces --
   ---------------------------------------

   function Evaluate_White_Unprotected_Pieces return Integer is
      Score : Integer := 0;
   begin
      -- Look for unprotected pieces
      for I in White_Pieces'Range loop
	 exit when White_Pieces (I) = No_Piece;
	 if White_Pieces (I).Piece /= White_King then
	    if not Find_Attack (White_Pieces (I).Square, White) then
	       Score := Score - Unprotected_Piece_Penalty;
	       if Game_Status = End_Game and then White_Pieces (I).Piece = White_Pawn then
		  Score := Score - Unprotected_Pawn_On_End_Game;
	       end if;
	    end if;
	 end if;
      end loop;
      return Score;
   end Evaluate_White_Unprotected_Pieces;


   ---------------------------------------
   -- Evaluate_Black_Unprotected_Pieces --
   ---------------------------------------

   function Evaluate_Black_Unprotected_Pieces return Integer is
      Score : Integer := 0;
   begin
      -- Look for unprotected pieces!
      for I in Black_Pieces'Range loop
	 exit when Black_Pieces (I) = No_Piece;
	 if Black_Pieces (I).Piece /= Black_King then
	    if not Find_Attack (Black_Pieces (I).Square, Black) then
	       Score := Score - Unprotected_Piece_Penalty;
	        if Game_Status = End_Game and then Black_Pieces (I).Piece = Black_Pawn then
		  Score := Score - Unprotected_Pawn_On_End_Game;
	       end if;
	    end if;
	 end if;
      end loop;
      return Score;
   end Evaluate_Black_Unprotected_Pieces;


   ---------------------------------------
   -- Evaluate_White_Material_Advantage --
   ---------------------------------------

   function Evaluate_White_Material_Advantage (Score : in Integer) return Integer is
   begin
      -- encourage capture when one side has material advantage
      if Count_White_Pieces >= Count_Black_Pieces and then Score > 80 then
	 return Material_Advantage_Recapture_Encourage;
      end if;
      return 0;
   end Evaluate_White_Material_Advantage;


   ---------------------------------------
   -- Evaluate_Black_Material_Advantage --
   ---------------------------------------

   function Evaluate_Black_Material_Advantage (Score : in Integer) return Integer is
   begin
      -- encourage capture when one side has material advantage
      if Count_Black_Pieces >= Count_White_Pieces and then Score > 80 then
	 return Material_Advantage_Recapture_Encourage;
      end if;
      return 0;
   end Evaluate_Black_Material_Advantage;


   --------------------------------
   -- Populate_Weak_Square_Board --
   --------------------------------

   procedure Populate_Weak_Square_Board is
      Pawn_File         : Integer := 0;
      Pawn_Rank         : Integer := 0;
      Count_Pawns       : Integer := 0;
      Square            : Integer;
   begin

       for I in 0 .. White_Pawn_Counter - 1 loop
	 Pawn_File := File (White_Pawn_Position (White_Pawn_Position'First + I)) + 1;
	 Pawn_Rank := White_Pawn_Rank (Pawn_File);
	 if Pawn_File = 1 then
	    if White_Pawn_Rank (2) > Pawn_Rank or else White_Pawn_Rank (2) = 0 then
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank - 1);
	       White_Weak_Board (Square) := True;
	    end if;
	 elsif Pawn_File = 8 then
	    if White_Pawn_Rank (7) > Pawn_Rank or else White_Pawn_Rank (7) = 0 then
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank - 1);
	       White_Weak_Board (Square) := True;
	    end if;
	 else
	    if (White_Pawn_Rank (Pawn_File + 1) > Pawn_Rank or else White_Pawn_Rank (Pawn_File + 1) = 0)
	      and then (White_Pawn_Rank (Pawn_File - 1) > Pawn_Rank or else White_Pawn_Rank (Pawn_File - 1) = 0)
	    then
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank - 1);
	       White_Weak_Board (Square) := True;
	    end if;
	 end if;
      end loop;
      for I in 0 .. Black_Pawn_Counter - 1 loop
	 Pawn_File := File (Black_Pawn_Position (Black_Pawn_Position'First + I)) + 1;
	 Pawn_Rank := Black_Pawn_Rank (Pawn_File);
--  	 Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank);
--  	 Put_Line ("Black pawn on " & Pc_Sqr (Square) & " on file" & Integer'Image (Pawn_File) & " and rank" & Integer'Image (Pawn_Rank));
	 if Pawn_File = 1 then
	    if Black_Pawn_Rank (2) < Pawn_Rank or else Black_Pawn_Rank (2) = 9 then
--  	       Square := 1 + 1 + Pawn_File + 12 * (10 - Pawn_Rank);
--  	       Black_Weak_Board (Square) := True;
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank + 1);
	       Black_Weak_Board (Square) := True;
	    end if;
	 elsif Pawn_File = 8 then
	     if Black_Pawn_Rank (7) < Pawn_Rank or else Black_Pawn_Rank (7) = 9 then
--  	       Square := 1 - 1 + Pawn_File + 12 * (10 - Pawn_Rank);
--  	       Black_Weak_Board (Square) := True;
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank + 1);
	       Black_Weak_Board (Square) := True;
	    end if;
	 else
	    if (Black_Pawn_Rank (Pawn_File + 1) < Pawn_Rank or else Black_Pawn_Rank (Pawn_File + 1 ) = 9)
	      and then (Black_Pawn_Rank (Pawn_File - 1) < Pawn_Rank or else Black_Pawn_Rank (Pawn_File - 1 ) = 9)
	    then
	       Square := 1 + Pawn_File + 12 * (10 - Pawn_Rank + 1);
	       Black_Weak_Board (Square) := True;
	    end if;
	 end if;
      end loop;
   end Populate_Weak_Square_Board;


   --------------
   -- Distance --
   --------------

   function Distance (From, To : in Integer) return Integer is
   begin
      return abs ((File (From) - File (To)) + (Rank (From) - Rank (To)));
   end Distance;


end ACEvaluate;
