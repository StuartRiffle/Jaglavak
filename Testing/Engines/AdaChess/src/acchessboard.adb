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


with Ada.Text_IO; 	use Ada.Text_IO;
with Ada.Directories; 	use Ada.Directories;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
with Ada.Strings; use Ada.Strings;
with ACHash; use ACHash;

with Ada.Exceptions; 		use Ada.Exceptions;


package body ACChessBoard is
   
   -------------
   -- MVV_LVA --
   -------------

   procedure MVV_LVA (Move : in out Move_Type) is
   begin
      if not Is_Capture (Move) then
	 Move.Score := History_Heuristic (Ply, Move.From, Move.To);
      else
	 Move.Score := MVV_LVA_Table (Move.Captured, Move.Piece);
      end if;
   end MVV_LVA;

   
   -------------------------
   -- Update_Killer_Moves --
   -------------------------
     
   procedure Update_Killer_Moves (Move : in Move_Type; Score : in Integer) is
   begin
      if Score > Killer_Score_1 (Ply) then
	 Killer_Score_3 (Ply) := Killer_Score_2 (Ply);
	 Killer_Score_2 (Ply) := Killer_Score_1 (Ply);
	 Killer_Score_1 (Ply) := Score;
	 Killer_Heuristic_3 (Ply) := Killer_Heuristic_2 (Ply);
	 Killer_Heuristic_2 (Ply) := Killer_Heuristic_1 (Ply);
	 Killer_Heuristic_1 (Ply) := Move;
      elsif Score > Killer_Score_2 (Ply) then
	 Killer_Score_3 (Ply) := Killer_Score_2 (Ply);
	 Killer_Score_2 (Ply) := Score;
	 Killer_Heuristic_3 (Ply) := Killer_Heuristic_2 (Ply);
	 Killer_Heuristic_2 (Ply) := Move;
      elsif Score > Killer_Score_3 (Ply) then
	 Killer_Score_3 (Ply) := Score;
	 Killer_Heuristic_3 (Ply) := Move;
      end if;
   end Update_Killer_Moves;

   
   ----------------------
   -- Parse_Input_Move --
   ----------------------
   
   function Parse_Input_Move (Input : in String) return Move_Type is
      M : Move_Type;
   begin
      M.From := 2 + Character'Pos (Input (Input'First)) - Character'Pos ('a') +
	12 * ( 10 - (Character'Pos (Input (Input'First + 1)) - Character'Pos ('0') ));
      M.To := 2 + Character'Pos (Input (Input'First + 2)) - Character'Pos ('a') +
	12 * ( 10 - (Character'Pos (Input (Input'First + 3)) - Character'Pos ('0') ));
      if Input'Length = 5 then
	 case Input (Input'First + 4) is
	    when 'n' =>
	       if Side_To_Move = White then
		  M.Promotion := White_Knight;
	       else
		  M.Promotion := Black_Knight;
	       end if;
	    when 'b' =>
	       if Side_To_Move = White then
		  M.Promotion := White_Bishop;
	       else
		  M.Promotion := Black_Bishop;
	       end if;
	    when 'r' =>
	       if Side_To_Move = White then
		  M.Promotion := White_Rook;
	       else
		  M.Promotion := Black_Rook;
	       end if;
	    when 'q' =>
	       if Side_To_Move = White then
		  M.Promotion := White_Queen;
	       else
		  M.Promotion := Black_Queen;
	       end if;
	    when others => null; -- illegal move
	 end case;
      end if;

      for I in Ply_Type'Range loop
	 exit when Moves_List (Ply, I) = No_Move;
	 if Moves_List (Ply, I).From = M.From
	   and then Moves_List (Ply, I).To = M.To
	   and then Moves_List (Ply, I).Promotion = M.Promotion then
	    return Moves_List (Ply, I);
	 end if;
      end loop;
      return No_Move;
   end Parse_Input_Move;


   ----------
   -- Echo --
   ----------
   
   procedure Echo (M : in Move_Type) is
      Letter        : array (0 .. 7) of Character := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' );
      Letter_Pieces : array (1 .. 11) of Character := ( ' ', 'p', 'p', 'n', 'n', 'b', 'b', 'r', 'r', 'q', 'q');
   begin
      if ChessBoard (M.From) = Frame then
	 return;
      end if;
      if File (M.From) not in Letter'Range then
	 return;
      end if;
      Put (Letter (File (M.From)));
      Put (Rank (M.From), 0);
      if M.To /= 0 then
	 Put (Letter (File (M.To)));
	 Put (Rank (M.To), 0);
      end if;
      if Is_Piece (M.Promotion) then
	 Put (Letter_Pieces (M.Promotion));
      end if;
   end Echo;

   
   -----------
   -- Echo2 --
   -----------
   
   function Echo2 (M : in Move_Type) return String is
      Output : String (1 .. 5) := (others => ' ');
      Length : Integer := 4;
      Sqr    : constant array (0 .. 143) of String (1 .. 2) :=
	( "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ",
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
   begin
      Output (1 .. 2) := Sqr (M.From);
      Output (3 .. 4) := Sqr (Integer (M.To));
      case M.Promotion is
	 when White_Knight | Black_Knight =>
	    Output (5) := 'n';
	    Length := 5;
	 when White_Bishop | Black_Bishop =>
	    Output (5) := 'b';
	    Length := 5;
	 when White_Rook | Black_Rook =>
	    Output (5) := 'r';
	    Length := 5;
	 when White_Queen | Black_Queen =>
	    Output (5) := 'q';
	    Length := 5;
	 when others => null;
	    Length := 4;
      end case;
      return Output (Output'First .. Length);
   end Echo2;

   
   ----------------------
   -- Reset_En_Passant --
   ----------------------
   
   procedure Reset_En_Passant is
   begin
      En_Passant := 0;
   end Reset_En_Passant;

   
   ----------
   -- Play --
   ----------
   
   function Play (Move : in Move_Type) return Boolean is
      Temp     : Move_Type;
      Piece    : Integer := ChessBoard (Move.From);
      From, To : Integer;
   begin

      if not Is_Valid (Move) then
	 return False;
      end if;

      Temp.From := Move.From;
      Temp.To := Move.To;
      Temp.Fifty := Fifty;
      Temp.Piece := Move.Piece;
      Temp.Captured := Move.Captured;
      Temp.Score := Move.Score;
      Temp.En_Passant := En_Passant;
      Temp.Promotion := Move.Promotion;
      Temp.Castle := Castle;
      Temp.Hash := Hash;
      Temp.Flags := Move.Flags;

      -- if it is a castle, then find the rook to move
      if Move.Flags.Castle then
	 case Move.To is
	    when G1 =>  -- white kingside castle
	       From := H1;
	       To := F1;
	       Castle.White_Kingside := False;
	       Update_White_Piece (From, To, White_Rook);
	    when C1 => -- white queenside castle
	       From := A1;
	       To := D1;
	       Castle.White_Queenside := False;
	       Update_White_Piece (From, To, White_Rook);
	    when G8 => -- black kingside castle
	       From := H8;
	       To := F8;
	       Castle.Black_Kingside := False;
	       Update_Black_Piece (From, To, Black_Rook);
	    when C8 => -- black queenside castle
	       From := A8;
	       To := D8;
	       Castle.Black_Queenside := False;
	       Update_Black_Piece (From, To, Black_Rook);
	    when others => null; -- raise CastleException
	 end case;
	 -- move the rook involved to the castle
	 Chessboard (To) := Chessboard (From);
	 ChessBoard (From) := Empty;
      end if;


      if Move.Piece = White_Pawn or else Move.Piece = Black_Pawn or else Is_Piece (Move.Captured) then
	 Fifty := 0;
      else
	 Fifty := Fifty + 1;
      end if;

      if Move.Flags.Pawn_Two_Square_Move then
	 if Side_To_Move = White then
	    En_Passant := Move.From + North;
	 else
	    En_Passant := Move.From + South;
	 end if;
      else
	 --Reset_En_Passant;
	 En_Passant := 0;
      end if;

      if Move.Flags.En_Passant then
	 if Side_To_Move = White then
	    ChessBoard (Move.To + South) := Empty;
	    Delete_Black_Piece (Move.To + South);
	 else
	    ChessBoard (Move.To + North) := Empty;
	    Delete_White_Piece (Move.To + North);
	 end if;
      end if;


      if Is_Piece (Move.Promotion) then
	 ChessBoard (Move.To) := Move.Promotion;
      else
	 ChessBoard (Move.To) := ChessBoard (Move.From);
      end if;
      ChessBoard (Move.From) := Empty;

      -- update any castle info (and king position)
      if Move.From = A8 or else Move.To = A8 then
	 Castle.Black_Queenside := False;
      elsif Move.From = H8 or else Move.To = H8 then
	 Castle.Black_Kingside := False;
      elsif Move.From = A1 or else Move.To = A1 then
	 Castle.White_Queenside := False;
      elsif Move.From = H1 or else Move.To = H1 then
	 Castle.White_Kingside := False;
      end if;

      if Move.Piece = White_King then
	 White_King_Position := Move.To;
	 Castle.White_Kingside := False;
	 Castle.White_Queenside := False;
      elsif Move.Piece = Black_King then
	 Black_King_Position := Move.To;
	 Castle.Black_Kingside := False;
	 Castle.Black_Queenside := False;
      end if;

      if Is_White (Move.Captured) then
	 Delete_White_Piece (Move.To);
      elsif Is_Black (Move.Captured) then
	 Delete_Black_Piece (Move.To);
      end if;

      if Is_White (Piece) then
	 if Is_Piece (Move.Promotion) then
	    Update_White_Piece (Move.From, Move.To, Move.Promotion);
	 else
	    Update_White_Piece (Move.From, Move.To, Piece);
	 end if;
      elsif Is_Black (Piece) then
	 if Is_Piece (Move.Promotion) then
	    Update_Black_Piece (Move.From, Move.To, Move.Promotion);
	 else
	    Update_Black_Piece (Move.From, Move.To, Piece);
	 end if;
      end if;


      Change_Side_To_Move;
      History_Moves (History_Ply) := Temp;
      Ply := Ply + 1;
      History_Ply := History_Ply + 1;

      Update_Hash;

      return True;
   end Play;


   ----------
   -- Undo --
   ----------

   procedure Undo is
      Move       : Move_Type;
      From, To   : Integer;
   begin

      --        pragma Assert (History_Ply >= Ply_Type'First, "Cannot Undo with History_Ply = " & Integer'Image (History_Ply));
      if History_Ply <= History_Started_At then
	 return; -- PreCondition
      end if;

      Ply := Ply - 1;
      History_Ply := History_Ply - 1;
      Move := History_Moves (History_Ply);
      Castle := Move.Castle;

      Change_Side_To_Move;

      Fifty := Move.Fifty;
      En_Passant := Move.En_Passant;
      Hash := Move.Hash;

      -- if it is a castle, then find the rook to move
      if Move.Flags.Castle then
	 -- king is_in_check() already checked
	 case Move.To is
	    when G1 =>  -- white kingside castle
	       From := F1;
	       To := H1;
	       Castle.White_Kingside := True;
	       Update_White_Piece (From, To, White_Rook);
	    when C1 =>
	       From := D1;
	       To := A1;
	       Castle.White_Queenside := True;
	       Update_White_Piece (From, To, White_Rook);
	    when G8 =>
	       From := F8;
	       To := H8;
	       Castle.Black_Kingside := True;
	       Update_Black_Piece (From, To, Black_Rook);
	    when C8 =>
	       From := D8;
	       To := A8;
	       Castle.Black_Queenside := True;
	       Update_Black_Piece (From, To, Black_Rook);
	    when others => null; -- raise CastleException
	 end case;
	 -- do the rook move
	 Chessboard (To) := Chessboard (From);
	 ChessBoard (From) := Empty;
      end if;


      if Move.Flags.En_Passant then
	 if Side_To_Move = White then
	    ChessBoard (En_Passant + South) := Black_Pawn;
	    Add_Black_Piece (En_Passant + South, Black_Pawn);
	 else
	    ChessBoard (En_Passant + North) := White_Pawn;
	    Add_White_Piece (En_Passant + North, White_Pawn);
	 end if;
      end if;


      if Is_Piece (Move.Promotion) then
	 if Side_To_Move = White then
	    ChessBoard (Move.From) := White_Pawn;
	 else
	    ChessBoard (Move.From) := Black_Pawn;
	 end if;
      else
	 ChessBoard (Move.From) := ChessBoard (Move.To);
      end if;
      ChessBoard (Move.To) := Move.Captured;


      if Is_White (Move.Piece) then
	 Update_White_Piece (Move.To, Move.From, Move.Piece);
      elsif Is_Black (Move.Piece) then
	 Update_Black_Piece (Move.To, Move.From, Move.Piece);
      end if;

      if Is_White (Move.Captured) then
	 Add_White_Piece (Move.To, Move.Captured);
      elsif Is_Black (Move.Captured) then
	 Add_Black_Piece (Move.To, Move.Captured);
      end if;

      if Move.Piece = White_King then
	 White_King_Position := Move.From;
      elsif Move.Piece = Black_King then
	 Black_King_Position := Move.From;
      end if;

      -- Clear previously history data. Just to avoid bugs and blunders!
      -- It is not mandatory but helps to prevent
      -- unpredicted data to be processed!
      History_Moves (History_Ply + 1) := No_Move;

   end Undo;


   ----------
   -- Undo --
   ----------

   procedure Undo (Times : in Natural) is
   begin
      for I in 1 .. Times loop
	 Undo;
      end loop;
   end Undo;

   --------------
   -- Is_Valid --
   --------------
   
   function Is_Valid (Move : in Move_Type) return Boolean is
   begin
      return Is_Piece (Move.Piece) and then Is_In_Board (Move.To);
   end Is_Valid;



   ----------------------------
   -- Generate_Capture_Moves --
   ----------------------------

   procedure Generate_Capture_Moves is
   begin
      Clear_Moves_List;

      if Side_To_Move = White then
	 for I in White_Pieces'Range loop
	    exit when White_Pieces (I) = No_Piece;

	    case White_Pieces (I).Piece is
	       when White_Pawn => Generate_White_Pawn_Capture_Moves (I, White_Pieces (I).Square);
	       when White_Knight => Generate_White_Knight_Capture_Moves (I, White_Pieces (I).Square);
	       when White_Bishop => Generate_White_Bishop_Capture_Moves (I, White_Pieces (I).Square);
	       when White_Rook => Generate_White_Rook_Capture_Moves (I, White_Pieces (I).Square);
	       when White_Queen => Generate_White_Queen_Capture_Moves (I, White_Pieces (I).Square);
	       when White_King => Generate_White_King_Capture_Moves (I, White_Pieces (I).Square);
	       when others => null; --raise exception
	    end case;

	 end loop;
      else -- blacks
	 for I in Black_Pieces'Range loop
	    exit when Black_Pieces (I) = No_Piece;

	    case Black_Pieces (I).Piece is
	       when Black_Pawn => Generate_Black_Pawn_Capture_Moves (I, Black_Pieces (I).Square);
	       when Black_Knight => Generate_Black_Knight_Capture_Moves (I, Black_Pieces (I).Square);
	       when Black_Bishop => Generate_Black_Bishop_Capture_Moves (I, Black_Pieces (I).Square);
	       when Black_Rook => Generate_Black_Rook_Capture_Moves (I, Black_Pieces (I).Square);
	       when Black_Queen => Generate_Black_Queen_Capture_Moves (I, Black_Pieces (I).Square);
	       when Black_King => Generate_Black_King_Capture_Moves (I, Black_Pieces (I).Square);
	       when others => null; -- raise exception
	    end case;

	 end loop;
      end if;
      
      Moves_List (Ply, Moves_Counter (Ply) + 1) := No_Move;
      
   end Generate_Capture_Moves;


   --------------------
   -- Generate_Moves --
   --------------------

   procedure Generate_Moves is
      Flags       : Flag_Type;
      Square      : Integer;
   begin
      Clear_Moves_List;

      if Side_To_Move = White then
	 Force_Test_Validity := Has_King_In_Check (White);
	 for I in White_Pieces'Range loop
	    exit when White_Pieces (I) = No_Piece;
	    Square := White_Pieces (I).Square;
	    case White_Pieces (I).Piece is
	       when White_Pawn => Generate_White_Pawn_Moves (I, Square);
	       when White_Knight => Generate_White_Knight_Moves (I, Square);
	       when White_Bishop => Generate_White_Bishop_Moves (I, Square);
	       when White_Rook => Generate_White_Rook_Moves (I, Square);
	       when White_Queen => Generate_White_Queen_Moves (I, Square);
	       when White_King => Generate_White_King_Moves (I, Square);
	       when others => null; --raise exception
	    end case;

	 end loop;

      else -- blacks
	 Force_Test_Validity := Has_King_In_Check (Black);
	 for I in Black_Pieces'Range loop
	    exit when Black_Pieces (I) = No_Piece;
	    Square := Black_Pieces (I).Square;

	    case Black_Pieces (I).Piece is
	       when Black_Pawn => Generate_Black_Pawn_Moves (I, Square);
	       when Black_Knight => Generate_Black_Knight_Moves (I, Square);
	       when Black_Bishop => Generate_Black_Bishop_Moves (I, Square);
	       when Black_Rook => Generate_Black_Rook_Moves (I, Square);
	       when Black_Queen => Generate_Black_Queen_Moves (I, Square);
	       when Black_King => Generate_Black_King_Moves (I, Square);
	       when others => null; -- raise exception
	    end case;

	 end loop;
      end if;

      -- now we have the attacks information, then look for castle moves!
      Reset (Flags);
      if Side_To_Move = White then
	 if Castle.White_Kingside then
	    if Is_Empty (F1) and then Is_Empty (G1) then
	       if Find_Attack (F1, Black) = False and then Has_King_In_Check (White) = False then -- and then Attack (G1) = No_Piece then
		  Flags.Castle := True;
		  Register_Move (From => E1, To => G1, Flags => Flags, Test_Validity => True);
	       end if;
	    end if;
	 end if;
	 if Castle.White_Queenside then
	    if Is_Empty (D1) and then Is_Empty (C1) and then Is_Empty (B1) then
	       if Find_Attack (D1, Black) = False and then Has_King_In_Check (White) = False then -- and then Attack (C1) = No_Piece then
		  Flags.Castle := True;
		  Register_Move (From => E1, To => C1, Flags => Flags, Test_Validity => True);
	       end if;
	    end if;
	 end if;
      else
	 if Castle.Black_Kingside then
	    if Is_Empty (F8) and then Is_Empty (G8) then
	       if Find_Attack (F8, White) = False and then Has_King_In_Check (Black) = False then -- and then Attack (G8) = No_Piece then
		  Flags.Castle := True;
		  Register_Move (From => E8, To => G8, Flags => Flags, Test_Validity => True);
	       end if;
	    end if;
	 end if;
	 if Castle.Black_Queenside then
	    if Is_Empty (D8) and then Is_Empty (C8) and then Is_Empty (B8) then
	       if Find_Attack (D8, White) = False and then Has_King_In_Check (Black) = False then -- and then Attack (C8) = No_Piece then
		  Flags.Castle := True;
		  Register_Move (From => E8, To => C8, Flags => Flags, Test_Validity => True);
	       end if;
	    end if;
	 end if;
      end if;
      
       Moves_List (Ply, Moves_Counter (Ply) + 1) := No_Move;

   end Generate_Moves;


   -------------------------------
   -- Generate_White_Pawn_Moves --
   -------------------------------
   
   procedure Generate_White_Pawn_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Flags         : Flag_Type;
      Test_Validity : Boolean;
   begin
      Reset (Flags);
      Flags.Pawn_Move := True;
      Target := Square + North;
      if Is_Empty (Target) then
	 if Target in A8 .. H8 then
	    Flags.Pawn_Promote := True;
	 end if;
	 Test_Validity := False;
	 if Rank (Square) = Rank (White_King_Position)
	   or else Diagonal (Square) = Diagonal (White_King_Position)
	   or else Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position) then
	    Test_Validity := True;
	 end if;
	 Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 if Square in A2 .. H2 then
	    Target := Target + North;
	    if Is_Empty (Target) then
	       Flags.Pawn_Two_Square_Move := True;
	       Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	       Flags.Pawn_Two_Square_Move := False;
	    end if;
	 end if;
      end if;
      -- take on left/right
      Target := Square + North_West;
      if Target in A8 .. H8 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True); -- foce test on en_passant
	    Flags.En_Passant := False;
	 elsif Is_Black (ChessBoard (Target)) then
	    Test_Validity := False;
	    if Rank (Square) = Rank (White_King_Position)
	      or else Diagonal (Square) = Diagonal (White_King_Position)
	      or else Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position)
	      or else File (Square) = File (White_King_Position) then
	       Test_Validity := True;
	    end if;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 end if;
      end if;
      Target := Square + North_East;
      if Target in A8 .. H8 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_Black (ChessBoard (Target)) then
	    Test_Validity := False;
	    if Rank (Square) = Rank (White_King_Position)
	      or else Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position)
	      or else Diagonal (Square) = Diagonal (White_King_Position)
	      or else File (Square) = File (White_King_Position) then
	       Test_Validity := True;
	    end if;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 end if;
      end if;
   end Generate_White_Pawn_Moves;


   ---------------------------------
   -- Generate_White_Knight_Moves --
   ---------------------------------
   
   procedure Generate_White_Knight_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Knight_Offsets'Range loop
	 Target := Square + Knight_Offsets (I);
	 if Is_In_Board (Target) then
	    if not Is_White (ChessBoard (Target)) then
	       Test_Validity := False;
	       if Rank (Square) = Rank (White_King_Position)
		 or else File (Square) = File (White_King_Position)
		 or else Diagonal (Square) = Diagonal (White_King_Position)
		 or else Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position) then
		  Test_Validity := True;
	       end if;
	       Register_Move (From  => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	 end if;
      end loop;
   end Generate_White_Knight_Moves;

   
   ---------------------------------
   -- Generate_White_Bishop_Moves --
   ---------------------------------

   procedure Generate_White_Bishop_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Bishop_Offsets'Range loop
	 Target := Square + Bishop_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_White (ChessBoard (Target)) then
	       Test_Validity := False;
	       if Rank (Square) = Rank (White_King_Position)
		 or else File (Square) = File (White_King_Position)
		 or else (Diagonal (Square) = Diagonal (White_King_Position) and then Diagonal (Target) /= Diagonal (Square))
		 or else (Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position) and then Anti_Diagonal (Target) /= Anti_Diagonal (Square))
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Bishop_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Bishop_Moves;


   -------------------------------
   -- Generate_White_Rook_Moves --
   -------------------------------
   
   procedure Generate_White_Rook_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Rook_Offsets'Range loop
	 Target := Square + Rook_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_White (ChessBoard (Target)) then
	       Test_Validity := False;
	       if (Rank (Square) = Rank (White_King_Position) and then Rank (Target) /= Rank (Square))
		 or else (File (Square) = File (White_King_Position) and then File (Target) /= File (Square))
		 or else Diagonal (Square) = Diagonal (White_King_Position)
		 or else Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position)
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Rook_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Rook_Moves;


   --------------------------------
   -- Generate_White_Queen_Moves --
   --------------------------------
     
   procedure Generate_White_Queen_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Queen_Offsets'Range loop
	 Target := Square + Queen_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_White (ChessBoard (Target)) then
	       Test_Validity := False;
	       if (Rank (Square) = Rank (White_King_Position) and then Rank (Target) /= Rank (Square))
		 or else (File (Square) = File (White_King_Position) and then File (Target) /= File (Square))
		 or else (Diagonal (Square) = Diagonal (White_King_Position) and then Diagonal (Target) /= Diagonal (Square))
		 or else (Anti_Diagonal (Square) = Anti_Diagonal (White_King_Position) and then Anti_Diagonal (Target) /= Anti_Diagonal (Square))
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Queen_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Queen_Moves;

   
   -------------------------------
   -- Generate_White_King_Moves --
   -------------------------------

   procedure Generate_White_King_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in King_Offsets'Range loop
	 Target := Square + King_Offsets (I);
	 if Is_In_Board (Target) then
	    if not Is_White (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	 end if;
      end loop;
   end Generate_White_King_Moves;

   
   -------------------------------
   -- Generate_Black_Pawn_Moves --
   -------------------------------

   procedure Generate_Black_Pawn_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Flags         : Flag_Type;
      Test_Validity : Boolean;
   begin
      Reset (Flags);
      Flags.Pawn_Move := True;
      Target := Square + South;
      if Is_Empty (Target) then
	 if Target in A1 .. H1 then
	    Flags.Pawn_Promote := True;
	 end if;
	 Test_Validity := False;
	 if Rank (Square) = Rank (Black_King_Position)
	   or else Diagonal (Square) = Diagonal (Black_King_Position)
	   or else Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position) then
	    Test_Validity := True;
	 end if;
	 Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 Flags.En_Passant := False;
	 if Square in A7 .. H7 then
	    Target := Target + South;
	    if Is_Empty (Target) then
	       Flags.Pawn_Two_Square_Move := True;
	       Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	       Flags.Pawn_Two_Square_Move := False;
	    end if;
	 end if;
      end if;
      -- take on left/right
      Target := Square + South_West;
      Test_Validity := False;
      if Target in A1 .. H1 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True); -- force test_validity on en-passant
	    Flags.En_Passant := False;
	 elsif Is_White (ChessBoard (Target)) then
	    if Rank (Square) = Rank (Black_King_Position)
	      or else Diagonal (Square) = Diagonal (Black_King_Position)
	      or else Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position) 
	      or else File (Square) = File (Black_King_Position) then
	       Test_Validity := True;
	    end if;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 end if;
      end if;
      Target := Square + South_East;
      Test_Validity := False;
      if Target in A1 .. H1 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_White (ChessBoard (Target)) then
	    if Rank (Square) = Rank (Black_King_Position)
	      or else Diagonal (Square) = Diagonal (Black_King_Position)
	      or else Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position)
	      or else File (Square) = File (Black_King_Position) then
	       Test_Validity := True;
	    end if;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => Test_Validity);
	 end if;
      end if;
   end Generate_Black_Pawn_Moves;

   
   ---------------------------------
   -- Generate_Black_Knight_Moves --
   ---------------------------------

   procedure Generate_Black_Knight_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Knight_Offsets'Range loop
	 Target := Square + Knight_Offsets (I);
	 if Is_In_Board (Target) then
	    if not Is_Black (ChessBoard (Target)) then
	       Test_Validity := False;
	       if Rank (Square) = Rank (Black_King_Position)
		 or else File (Square) = File (Black_King_Position)
		 or else Diagonal (Square) = Diagonal (Black_King_Position)
		 or else Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position) then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	 end if;
      end loop;
   end Generate_Black_Knight_Moves;

   
   ---------------------------------
   -- Generate_Black_Bishop_Moves --
   ---------------------------------

   procedure Generate_Black_Bishop_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Bishop_Offsets'Range loop
	 Target := Square + Bishop_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_Black (ChessBoard (Target)) then
	       Test_Validity := False;
	       if Rank (Square) = Rank (Black_King_Position)
		 or else File (Square) = File (Black_King_Position)
		 or else (Diagonal (Square) = Diagonal (Black_King_Position) and then Diagonal (Target) /= Diagonal (Square))
		 or else (Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position) and then Anti_Diagonal (Target) /= Anti_Diagonal (Square))
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Bishop_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Bishop_Moves;
   

   -------------------------------
   -- Generate_Black_Rook_Moves --
   -------------------------------

   procedure Generate_Black_Rook_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Rook_Offsets'Range loop
	 Target := Square + Rook_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_Black (ChessBoard (Target)) then
	       Test_Validity := False;
	       if (Rank (Square) = Rank (Black_King_Position) and then Rank (Target) /= Rank (Square))
		 or else (File (Square) = File (Black_King_Position) and then File (Target) /= File (Square))
		 or else Diagonal (Square) = Diagonal (Black_King_Position)
		 or else Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position)
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Rook_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Rook_Moves;
   

   --------------------------------
   -- Generate_Black_Queen_Moves --
   --------------------------------

   procedure Generate_Black_Queen_Moves (Index, Square : in Integer) is
      Target        : Integer;
      Test_Validity : Boolean;
   begin
      for I in Queen_Offsets'Range loop
	 Target := Square + Queen_Offsets (I);
	 while Is_In_Board (Target) loop
	    if not Is_Black (ChessBoard (Target)) then
	       Test_Validity := False;
	       if (Rank (Square) = Rank (Black_King_Position) and then Rank (Target) /= Rank (Square))
		 or else (File (Square) = File (Black_King_Position) and then File (Target) /= File (Square))
		 or else (Diagonal (Square) = Diagonal (Black_King_Position) and then Diagonal (Target) /= Diagonal (Square))
		 or else (Anti_Diagonal (Square) = Anti_Diagonal (Black_King_Position) and then Anti_Diagonal (Target) /= Anti_Diagonal (Square))
	       then
		  Test_Validity := True;
	       end if;
	       Register_Move (From => Square, To => Target, Test_Validity => Test_Validity);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Queen_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Queen_Moves;

   
   -------------------------------
   -- Generate_Black_King_Moves --
   -------------------------------

   procedure Generate_Black_King_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in King_Offsets'Range loop
	 Target := Square + King_Offsets (I);
	 if Is_In_Board (Target) then
	    if not Is_Black (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	 end if;
      end loop;
   end Generate_Black_King_Moves;


   ---------------------------------------
   -- Generate_White_Pawn_Capture_Moves --
   ---------------------------------------

   procedure Generate_White_Pawn_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
      Flags  : Flag_Type;
   begin
      Reset (Flags);
      Flags.Pawn_Move := True;
      Target := Square + North;
      if Target in A8 .. H8 and then Is_Empty (Target) then
	 Flags.Pawn_Promote := True;
	 Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 Flags.Pawn_Promote := False;
      end if;
      -- take on left/right
      Target := Square + North_West;
      if Target in A8 .. H8 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_Black (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 end if;
      end if;
      Target := Square + North_East;
      if Target in A8 .. H8 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_Black (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 end if;
      end if;
   end Generate_White_Pawn_Capture_Moves;

   
   -----------------------------------------
   -- Generate_White_Knight_Capture_Moves --
   -----------------------------------------

   procedure Generate_White_Knight_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Knight_Offsets'Range loop
	 Target := Square + Knight_Offsets (I);
	 if Is_Black (ChessBoard (Target)) then
	    Register_Move (From  => Square, To => Target, Test_Validity => True);
	 end if;
      end loop;
   end Generate_White_Knight_Capture_Moves;


   -----------------------------------------
   -- Generate_White_Bishop_Capture_Moves --
   -----------------------------------------
   
   procedure Generate_White_Bishop_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Bishop_Offsets'Range loop
	 Target := Square + Bishop_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_Black (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Bishop_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Bishop_Capture_Moves;


   ---------------------------------------
   -- Generate_White_Rook_Capture_Moves --
   ---------------------------------------
   
   procedure Generate_White_Rook_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Rook_Offsets'Range loop
	 Target := Square + Rook_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_Black (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Rook_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Rook_Capture_Moves;

   
   ----------------------------------------
   -- Generate_White_Queen_Capture_Moves --
   ----------------------------------------

   procedure Generate_White_Queen_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Queen_Offsets'Range loop
	 Target := Square + Queen_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_Black (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Queen_Offsets (I);
	 end loop;
      end loop;
   end Generate_White_Queen_Capture_Moves;


   ---------------------------------------
   -- Generate_White_King_Capture_Moves --
   ---------------------------------------
   
   procedure Generate_White_King_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in King_Offsets'Range loop
	 Target := Square + King_Offsets (I);
	 if Is_Black (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Test_Validity => True);
	 end if;
      end loop;
   end Generate_White_King_Capture_Moves;


   ---------------------------------------
   -- Generate_Black_Pawn_Capture_Moves --
   ---------------------------------------

   procedure Generate_Black_Pawn_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
      Flags  : Flag_Type;
   begin
      Reset (Flags);
      Flags.Pawn_Move := True;
      Target := Square + South;
      if Target in A1 .. H1 and then Is_Empty (Target) then
	 Flags.Pawn_Promote := True;
	 Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 Flags.Pawn_Promote := False;
      end if;
      Flags.En_Passant := False;
      -- take on left/right
      Target := Square + South_West;
      if Target in A1 .. H1 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_White (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 end if;
      end if;
      Target := Square + South_East;
      if Target in A1 .. H1 then
	 Flags.Pawn_Promote := True;
      end if;
      if Is_In_Board (Target) then
	 if Target = En_Passant then
	    Flags.En_Passant := True;
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	    Flags.En_Passant := False;
	 elsif Is_White (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Flags => Flags, Test_Validity => True);
	 end if;
      end if;
   end Generate_Black_Pawn_Capture_Moves;


   -----------------------------------------
   -- Generate_Black_Knight_Capture_Moves --
   -----------------------------------------
   
   procedure Generate_Black_Knight_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Knight_Offsets'Range loop
	 Target := Square + Knight_Offsets (I);
	 if Is_White (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Test_Validity => True);
	 end if;
      end loop;
   end Generate_Black_Knight_Capture_Moves;


   -----------------------------------------
   -- Generate_Black_Bishop_Capture_Moves --
   -----------------------------------------
   
   procedure Generate_Black_Bishop_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Bishop_Offsets'Range loop
	 Target := Square + Bishop_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_White (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Bishop_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Bishop_Capture_Moves;


   ---------------------------------------
   -- Generate_Black_Rook_Capture_Moves --
   ---------------------------------------
   
   procedure Generate_Black_Rook_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Rook_Offsets'Range loop
	 Target := Square + Rook_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_White (ChessBoard (Target)) then
	       Register_Move ( From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Rook_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Rook_Capture_Moves;

   
   ----------------------------------------
   -- Generate_Black_Queen_Capture_Moves --
   ----------------------------------------

   procedure Generate_Black_Queen_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in Queen_Offsets'Range loop
	 Target := Square + Queen_Offsets (I);
	 while Is_In_Board (Target) loop
	    if Is_White (ChessBoard (Target)) then
	       Register_Move (From => Square, To => Target, Test_Validity => True);
	    end if;
	    exit when not Is_Empty (Target);
	    Target := Target + Queen_Offsets (I);
	 end loop;
      end loop;
   end Generate_Black_Queen_Capture_Moves;


   ---------------------------------------
   -- Generate_Black_King_Capture_Moves --
   ---------------------------------------
   
   procedure Generate_Black_King_Capture_Moves (Index, Square : in Integer) is
      Target : Integer;
   begin
      for I in King_Offsets'Range loop
	 Target := Square + King_Offsets (I);
	 if Is_White (ChessBoard (Target)) then
	    Register_Move (From => Square, To => Target, Test_Validity => True);
	 end if;
      end loop;
   end Generate_Black_King_Capture_Moves;


   -----------------------
   -- Has_King_In_Check --
   -----------------------

   function Has_King_In_Check (Side : in Integer) return Boolean is
      Target : Integer;
   begin
      if Side = White then
	 -- look for knights attacking king
	 for I in Knight_Offsets'Range loop
	    if Chessboard (White_King_Position + Knight_Offsets (I)) = Black_Knight then
	       return True;
	    end if;
	 end loop;
	 -- look for bishop/queen/king on diagonal
	 for I in Bishop_Offsets'Range loop
	    Target := White_King_Position + Bishop_Offsets (I);
	    if Target = Black_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Bishop_Offsets (I);
	    end loop;
	    if Chessboard (Target) = Black_Bishop or else Chessboard (Target) = Black_Queen then
	       return True;
	    end if;
	 end loop;
	 -- look for rook/queen/king on file/rank
	 for I in Rook_Offsets'Range loop
	    Target := White_King_Position + Rook_Offsets (I);
	    if Target = Black_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Rook_Offsets (I);
	    end loop;
	    if Chessboard (Target) = Black_Rook or else Chessboard (Target) = Black_Queen then
	       return True;
	    end if;
	 end loop;
	 -- pawn
	 if Chessboard (White_King_Position + North_West) = Black_Pawn then
	    return True;
	 end if;
	 if Chessboard (White_King_Position + North_East) = Black_Pawn then
	    return True;
	 end if;
	 return False;
      else
	 -- look for knights attacking king
	 for I in Knight_Offsets'Range loop
	    if Chessboard (Black_King_Position + Knight_Offsets (I)) = White_Knight then
	       return True;
	    end if;
	 end loop;
	 -- look for bishop/queen/king on diagonal
	 for I in Bishop_Offsets'Range loop
	    Target := Black_King_Position + Bishop_Offsets (I);
	    if Target = White_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Bishop_Offsets (I);
	    end loop;
	    if Chessboard (Target) = White_Bishop or else Chessboard (Target) = White_Queen then
	       return True;
	    end if;
	 end loop;
	 -- look for rook/queen/king on file/rank
	 for I in Rook_Offsets'Range loop
	    Target := Black_King_Position + Rook_Offsets (I);
	    if Target = White_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Rook_Offsets (I);
	    end loop;
	    if Chessboard (Target) = White_Rook or else Chessboard (Target) = White_Queen then
	       return True;
	    end if;
	 end loop;
	 -- pawn
	 if Chessboard (Black_King_Position + South_West) = White_Pawn then
	    return True;
	 end if;
	 if Chessboard (Black_King_Position + South_East) = White_Pawn then
	    return True;
	 end if;
	 return False;
      end if;
   end Has_King_In_Check;

   
   -----------------
   -- Find_Attack --
   -----------------

   function Find_Attack (Target_Square, Side : in Integer) return Boolean is
      Target : Integer;
   begin
      if Side = Black then
	 -- look for knights attacking Target_Square
	 for I in Knight_Offsets'Range loop
	    if Chessboard (Target_Square + Knight_Offsets (I)) = Black_Knight then
	       return True;
	    end if;
	 end loop;
	 -- look for bishop/queen/king on diagonal
	 for I in Bishop_Offsets'Range loop
	    Target := Target_Square + Bishop_Offsets (I);
	    if Target = Black_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Bishop_Offsets (I);
	    end loop;
	    if Chessboard (Target) = Black_Bishop or else Chessboard (Target) = Black_Queen then
	       return True;
	    end if;
	 end loop;
	 -- look for rook/queen/king on file/rank
	 for I in Rook_Offsets'Range loop
	    Target := Target_Square + Rook_Offsets (I);
	    if Target = Black_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Rook_Offsets (I);
	    end loop;
	    if Chessboard (Target) = Black_Rook or else Chessboard (Target) = Black_Queen then
	       return True;
	    end if;
	 end loop;
	 -- pawn
	 if Chessboard (Target_Square + North_West) = Black_Pawn then
	    return True;
	 end if;
	 if Chessboard (Target_Square + North_East) = Black_Pawn then
	    return True;
	 end if;
	 return False;
      else
	 -- look for knights attacking king
	 for I in Knight_Offsets'Range loop
	    if Chessboard (Target_Square + Knight_Offsets (I)) = White_Knight then
	       return True;
	    end if;
	 end loop;
	 -- look for bishop/queen/king on diagonal
	 for I in Bishop_Offsets'Range loop
	    Target := Target_Square + Bishop_Offsets (I);
	    if Target = White_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Bishop_Offsets (I);
	    end loop;
	    if Chessboard (Target) = White_Bishop or else Chessboard (Target) = White_Queen then
	       return True;
	    end if;
	 end loop;
	 -- look for rook/queen/king on file/rank
	 for I in Rook_Offsets'Range loop
	    Target := Target_Square + Rook_Offsets (I);
	    if Target = White_King_Position then
	       return True;
	    end if;
	    while Is_Empty (Target) loop
	       Target := Target + Rook_Offsets (I);
	    end loop;
	    if Chessboard (Target) = White_Rook or else Chessboard (Target) = White_Queen then
	       return True;
	    end if;
	 end loop;
	 -- pawn
	 if Chessboard (Target_Square + South_West) = White_Pawn then
	    return True;
	 end if;
	 if Chessboard (Target_Square + South_East) = White_Pawn then
	    return True;
	 end if;
	 return False;
      end if;
   end Find_Attack;


   ----------------------
   -- Clear_Moves_List --
   ----------------------
   
   procedure Clear_Moves_List is
   begin
--        for I in 1 .. Moves_Counter (Ply) loop
--  	 Moves_List (Ply, I) := No_Move;
--        end loop;
      Moves_Counter (Ply) := 0;
   end Clear_Moves_List;


   -------------------
   -- Register_Move --
   -------------------
   
   procedure Register_Move (From, To : in Integer; Test_Validity : Boolean) is
      Move : Move_Type := No_Move;
   begin
      Move.From := From;
      Move.To := To;
      Move.Piece := ChessBoard (From);
      Move.Captured := ChessBoard (To);
      Register_Move (Move, Test_Validity);
   end Register_Move;


   -------------------
   -- Register_Move --
   -------------------

   procedure Register_Move (From, To : in Integer; Flags : in Flag_Type; Test_Validity : Boolean) is
      Move : Move_Type := No_Move;
   begin
      Move.From := From;
      Move.To := To;
      Move.Piece := ChessBoard (From);
      Move.Captured := ChessBoard (To);
      Move.Flags := Flags;
      if Move.Flags.Pawn_Promote then
	 if Side_To_Move = White then
	    Move.Promotion := White_Knight;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := White_Bishop;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := White_Rook;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := White_Queen;
	    Register_Move (Move, Test_Validity);
	 else
	    Move.Promotion := Black_Knight;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := Black_Bishop;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := Black_Rook;
	    Register_Move (Move, Test_Validity);
	    Move.Promotion := Black_Queen;
	    Register_Move (Move, Test_Validity);
	 end if;
      else
	 Register_Move (Move, Test_Validity);
      end if;
   end Register_Move;

   
   -------------------
   -- Register_Move --
   -------------------

   procedure Register_Move (Move : in Move_Type; Test_Validity : in Boolean) is
   begin
      if Force_Test_Validity or else Test_Validity then
	 if Play (Move) then
	    if Has_King_In_Check (Opponent) then
	       Undo;
	       return;
	    end if;
	    Undo;
	    pragma Assert (Moves_Counter (Ply) < Ply_Type'Last, "Ply_Type fixed limit to " & Integer'Image (Ply_Type'Last) & " needs to be expanded|");
	    Moves_List (Ply, Ply_Type'First + Moves_Counter (Ply)) := Move;
	    Moves_Counter (Ply) := Moves_Counter (Ply) + 1;
	 end if;
      else
	 pragma Assert (Moves_Counter (Ply) < Ply_Type'Last, "Ply_Type fixed limit to " & Integer'Image (Ply_Type'Last) & " needs to be expanded|");
	 Moves_List (Ply, Ply_Type'First + Moves_Counter (Ply)) := Move;
	 Moves_Counter (Ply) := Moves_Counter (Ply) + 1;
      end if;
   end Register_Move;


   ----------
   -- Rank --
   ----------

   function Rank (Square : in Integer) return Integer is
   begin
      pragma Assert (Ranks (Square) /= -1, "Invalid Rank asked for square " & Integer'Image (Square) & " => " & Pc_Sqr (Square));
      return Ranks (Square);
   end Rank;

   ----------
   -- File --
   ----------

   function File (Square : in Integer) return Integer is
   begin
      pragma Assert (Files (Square) /= -1, "Invalid File asked for square" & Integer'Image (Square) & " => " & Pc_Sqr (Square));
      return Files (Square);
   end File;


   --------------
   -- Diagonal --
   --------------
   
   function Diagonal (Square : in Integer) return Integer is
   begin
      pragma Assert (Diagonals (Square) /= -1, "Invalid Diagonal asked for square" & Integer'Image (Square) & " => " & Pc_Sqr (Square));
      return Diagonals (Square);
   end Diagonal;
   
   -------------------
   -- Anti_Diagonal --
   -------------------

   function Anti_Diagonal (Square : in Integer) return Integer is
   begin
      pragma Assert (Anti_Diagonals (Square) /= -1, "Invalid Anti_Diagonals asked for square" & Integer'Image (Square) & " => " & Pc_Sqr (Square));
      return Anti_Diagonals (Square);
   end Anti_Diagonal;


   --------------
   -- Is_White --
   --------------

   function Is_White (Piece : in Integer) return Boolean is
   begin
      return Piece in White_Pawn | White_Knight | White_Bishop | White_Rook | White_Queen | White_King;
   end Is_White;


   --------------
   -- Is_Black --
   --------------
   
   function Is_Black (Piece : in Integer) return Boolean is
   begin
      return Piece in Black_Pawn | Black_Knight | Black_Bishop | Black_Rook | Black_Queen | Black_King;
   end Is_Black;

   
   -----------------
   -- Is_In_Board --
   -----------------

   function Is_In_Board (Square : in Integer) return Boolean is
   begin
      pragma Assert (Square in ChessBoard'Range, "Given Square outside ChessBoard'Range!");
      return ChessBoard (Square) /= Frame;
   end Is_In_Board;


   --------------
   -- Is_Piece --
   --------------

   function Is_Piece (Piece : in Integer) return Boolean is
   begin
      return Piece in White_Pawn | White_Knight | White_Bishop | White_Rook | White_Queen | White_King | Black_Pawn | Black_Knight | Black_Bishop | Black_Rook | Black_Queen | Black_King;
   end Is_Piece;

   
   --------------
   -- Is_Empty --
   --------------

   function Is_Empty (Square : in Integer) return Boolean is
   begin
      pragma Assert (Square in ChessBoard'Range, "Square" & Integer'Image (Square) & " is out of ChessBoard data type range");
      return ChessBoard (Square) = Empty;
   end Is_Empty;


   ----------------
   -- Is_Capture --
   ----------------

   function Is_Capture (Move : in Move_Type) return Boolean is
   begin
      pragma Assert (Move /= No_Move, "Cannot check for capture if 'Move' is not a valid move!");
      return Is_Piece (Move.Captured);
   end Is_Capture;

   
   ----------------------
   -- Last_Move_Played --
   ----------------------

   function Last_Move_Played return Move_Type is
   begin
      --        pragma Assert (History_Ply > 1, "No move has been played!");
      if History_Ply <= 1 then
	 return No_Move;
      end if;
      return History_Moves (History_Ply - 1);
   end Last_Move_Played;

   ------------------------------------
   ------------------------------------

   --     function Has_Moved_Pawn (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Pawn_Move) /= 0;
   --     end Has_Moved_Pawn;
   --
   --     ------------------------------------
   --
   --     function Has_Moved_Pawn_Two_Square (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Pawn_Move_Two_Square) /= 0;
   --     end Has_Moved_Pawn_Two_Square;
   --
   --     ------------------------------------
   --
   --     function Has_Captured (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Capture) /= 0;
   --     end Has_Captured;
   --
   --     ------------------------------------
   --
   --     function Has_Castled (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Castle) /= 0;
   --     end Has_Castled;
   --
   --     ------------------------------------
   --
   --     function Has_Castled_Kingside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Kingside_Castle) /= 0;
   --     end Has_Castled_Kingside;
   --
   --     ------------------------------------
   --
   --     function Has_Castled_Queenside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Queenside_Castle) /= 0;
   --     end Has_Castled_Queenside;
   --
   --     ------------------------------------
   --
   --     function Has_En_Passant (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_En_Passant) /= 0;
   --     end Has_En_Passant;
   --
   --     ------------------------------------
   --
   --     function Has_Promote (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Promotion) /= 0;
   --     end Has_Promote;
   --
   --     ------------------------------------
   --
   --     function White_Can_Castle_Kingside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_White_Can_Castle_Kingside) /= 0;
   --     end White_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     function White_Can_Castle_Queenside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_White_Can_Castle_Queenside) /= 0;
   --     end White_Can_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     function Black_Can_Castle_Kingside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Black_Can_Castle_Kingside) /= 0;
   --     end Black_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     function Black_Can_Castle_Queenside (Flag : in Flag_Type) return Boolean is
   --     begin
   --        return (Flag and Flag_Black_Can_Castle_Queenside) /= 0;
   --     end Black_Can_Castle_Queenside;
   --
   --     ------------------------------------
   --     ------------------------------------
   --
   --     procedure Set_On_Moved_Pawn (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Pawn_Move;
   --     end Set_On_Moved_Pawn;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Moved_Pawn_Two_Square (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Pawn_Move or Flag_Pawn_Move_Two_Square; -- anyway, force pawn move to be activate
   --     end Set_On_Moved_Pawn_Two_Square;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Captured (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Capture;
   --     end Set_On_Captured;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Castled (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Castle;
   --     end Set_On_Castled;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Castle_Kingside (Flag : out Flag_Type) is
   --     begin
   --        Set_On_Castled (Flag);
   --        Flag := Flag or Flag_Kingside_Castle;
   --     end Set_On_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Set_On_Castled (Flag);
   --        Flag := Flag or Flag_Queenside_Castle;
   --     end Set_On_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_White_Can_Castle_Kingside (Flag : out Flag_Type)is
   --     begin
   --        Flag := Flag or Flag_White_Can_Castle_Kingside;
   --     end Set_On_White_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_White_Can_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_White_Can_Castle_Queenside;
   --     end Set_On_White_Can_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Black_Can_Castle_Kingside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Black_Can_Castle_Kingside;
   --     end Set_On_Black_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Black_Can_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Black_Can_Castle_Queenside;
   --     end Set_On_Black_Can_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_En_Passant (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_En_Passant;
   --     end Set_On_En_Passant;
   --
   --     ------------------------------------
   --
   --     procedure Set_On_Promote (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag or Flag_Promotion;
   --     end Set_On_Promote;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Moved_Pawn (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Pawn_Move;
   --     end Set_Off_Moved_Pawn;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Moved_Pawn_Two_Square (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Pawn_Move_Two_Square;
   --     end Set_Off_Moved_Pawn_Two_Square;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Captured (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Capture;
   --     end Set_Off_Captured;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Castled (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Castle;
   --     end Set_Off_Castled;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Castle_Kingside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Kingside_Castle;
   --     end Set_Off_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Queenside_Castle;
   --     end Set_Off_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_En_Passant (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_En_Passant;
   --     end Set_Off_En_Passant;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Promote (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Promotion;
   --     end Set_Off_Promote;
   --
   --     ------------------------------------
   --     procedure Set_Off_White_Can_Castle_Kingside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_White_Can_Castle_Kingside;
   --     end Set_Off_White_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_White_Can_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_White_Can_Castle_Queenside;
   --     end Set_Off_White_Can_Castle_Queenside;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Black_Can_Castle_Kingside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Black_Can_Castle_Kingside;
   --     end Set_Off_Black_Can_Castle_Kingside;
   --
   --     ------------------------------------
   --
   --     procedure Set_Off_Black_Can_Castle_Queenside (Flag : out Flag_Type) is
   --     begin
   --        Flag := Flag and not Flag_Black_Can_Castle_Queenside;
   --     end Set_Off_Black_Can_Castle_Queenside;

   ------------------------------------


   ----------------
   -- Initialize --
   ----------------
   
   procedure Initialize is
   begin
      ChessBoard :=
	(Frame, Frame, Frame, Frame,   Frame,   Frame,  Frame, Frame,   Frame,   Frame, Frame, Frame,
  Frame, Frame, Frame, Frame,   Frame,   Frame,  Frame, Frame,   Frame,   Frame, Frame, Frame,
  Frame, Frame, Black_Rook, Black_Knight, Black_Bishop, Black_Queen, Black_King, Black_Bishop, Black_Knight, Black_Rook, Frame, Frame,
  Frame, Frame, Black_Pawn, Black_Pawn,   Black_Pawn,   Black_Pawn,  Black_Pawn, Black_Pawn,   Black_Pawn,   Black_Pawn, Frame, Frame,
  Frame, Frame, Empty, Empty,   Empty,   Empty,  Empty, Empty,   Empty,   Empty, Frame, Frame,
  Frame, Frame, Empty, Empty,   Empty,   Empty,  Empty, Empty,   Empty,   Empty, Frame, Frame,
  Frame, Frame, Empty, Empty,   Empty,   Empty,  Empty, Empty,   Empty,   Empty, Frame, Frame,
  Frame, Frame, Empty, Empty,   Empty,   Empty,  Empty, Empty,   Empty,   Empty, Frame, Frame,
  Frame, Frame, White_Pawn, White_Pawn,   White_Pawn,   White_Pawn,  White_Pawn, White_Pawn,   White_Pawn,   White_Pawn, Frame, Frame,
  Frame, Frame, White_Rook, White_Knight, White_Bishop, White_Queen, White_King, White_Bishop, White_Knight, White_Rook, Frame, Frame,
  Frame, Frame, Frame, Frame,   Frame,   Frame,  Frame, Frame,   Frame,   Frame, Frame, Frame,
  Frame, Frame, Frame, Frame,   Frame,   Frame,  Frame, Frame,   Frame,   Frame, Frame, Frame);
      

      White_Pieces_Counter := 0;
      Black_Pieces_Counter := 0;
      Reset_Piece_Table;
      -- inizializza i pezzi bianchi e neri nelle loro case
      Add_White_Piece (A1, White_Rook);
      Add_White_Piece (B1, White_Knight);
      Add_White_Piece (C1, White_Bishop);
      Add_White_Piece (D1, White_Queen);
      Add_White_Piece (E1, White_King);
      Add_White_Piece (F1, White_Bishop);
      Add_White_Piece (G1, White_Knight);
      Add_White_Piece (H1, White_Rook);
      Add_White_Piece (A2, White_Pawn);
      Add_White_Piece (B2, White_Pawn);
      Add_White_Piece (C2, White_Pawn);
      Add_White_Piece (D2, White_Pawn);
      Add_White_Piece (E2, White_Pawn);
      Add_White_Piece (F2, White_Pawn);
      Add_White_Piece (G2, White_Pawn);
      Add_White_Piece (H2, White_Pawn);
      --Now for black pieces
      Add_Black_Piece (A8, Black_Rook);
      Add_Black_Piece (B8, Black_Knight);
      Add_Black_Piece (C8, Black_Bishop);
      Add_Black_Piece (D8, Black_Queen);
      Add_Black_Piece (E8, Black_King);
      Add_Black_Piece (F8, Black_Bishop);
      Add_Black_Piece (G8, Black_Knight);
      Add_Black_Piece (H8, Black_Rook);
      Add_Black_Piece (A7, Black_Pawn);
      Add_Black_Piece (B7, Black_Pawn);
      Add_Black_Piece (C7, Black_Pawn);
      Add_Black_Piece (D7, Black_Pawn);
      Add_Black_Piece (E7, Black_Pawn);
      Add_Black_Piece (F7, Black_Pawn);
      Add_Black_Piece (G7, Black_Pawn);
      Add_Black_Piece (H7, Black_Pawn);

      Side_To_Move := White;
      for I in Ply_Type'Range loop
	 for J in Ply_Type'Range loop
	    Moves_List (I, J) := No_Move;
	 end loop;
      end loop;
      Clear_Moves_List;
      Moves_Counter := (others => 0);
      White_King_Position := E1;
      Black_King_Position := E8;
      Ply := Ply_Type'First;
      History_Ply := Ply_Type'First;
      History_Started_At := Ply_Type'First;
      Reset_En_Passant;
      Fifty := 0;
      Castle := (True, True, True, True);
      -- zobrist hash
      Initialize_Hash;
      Engine := Empty;

      for I in Ply_Type'Range loop
	 History_Moves (I) := No_Move;
      end loop;


      -- initialize then MMV_LVA value
      -- 1) Pawn
      -- 2) Knight
      -- 3) Bishop
      -- 4) Rook
      -- 5) Queen
      -- 10) King
      -- score := (10 * captured) + capturing
      for Captured in Piece_Type'Range loop
	 for Capturing in Piece_Type'Range loop
	    case Capturing is
	       when White_Pawn | Black_Pawn =>
		  case Captured is
		     when White_Pawn | Black_Pawn => MVV_LVA_Table (Captured, Capturing) := 11;
		     when White_Knight | Black_Knight =>  MVV_LVA_Table (Captured, Capturing) := 21;
		     when White_Bishop | Black_Bishop => MVV_LVA_Table (Captured, Capturing) := 31;
		     when White_Rook | Black_Rook => MVV_LVA_Table (Captured, Capturing) := 41;
		     when White_Queen | Black_Queen => MVV_LVA_Table (Captured, Capturing) := 51;
		     when White_King | Black_King => MVV_LVA_Table (Captured, Capturing) := 61;
		     when others => MVV_LVA_Table (Captured, Capturing) := 0;
		  end case;
	       when White_Knight | Black_Knight =>
		  case Captured is
		     when White_Pawn | Black_Pawn => MVV_LVA_Table (Captured, Capturing) := 12;
		     when White_Knight | Black_Knight =>  MVV_LVA_Table (Captured, Capturing) := 22;
		     when White_Bishop | Black_Bishop => MVV_LVA_Table (Captured, Capturing) := 32;
		     when White_Rook | Black_Rook => MVV_LVA_Table (Captured, Capturing) := 42;
		     when White_Queen | Black_Queen => MVV_LVA_Table (Captured, Capturing) := 52;
		     when White_King | Black_King => MVV_LVA_Table (Captured, Capturing) := 62;
		     when others => MVV_LVA_Table (Captured, Capturing) := 0;
		  end case;
	       when White_Bishop | Black_Bishop =>
		  case Captured is
		     when White_Pawn | Black_Pawn => MVV_LVA_Table (Captured, Capturing) := 13;
		     when White_Knight | Black_Knight =>  MVV_LVA_Table (Captured, Capturing) := 23;
		     when White_Bishop | Black_Bishop => MVV_LVA_Table (Captured, Capturing) := 33;
		     when White_Rook | Black_Rook => MVV_LVA_Table (Captured, Capturing) := 43;
		     when White_Queen | Black_Queen => MVV_LVA_Table (Captured, Capturing) := 53;
		     when White_King | Black_King => MVV_LVA_Table (Captured, Capturing) := 63;
		     when others => MVV_LVA_Table (Captured, Capturing) := 0;
		  end case;
	       when White_Rook | Black_Rook =>
		  case Captured is
		     when White_Pawn | Black_Pawn => MVV_LVA_Table (Captured, Capturing) := 14;
		     when White_Knight | Black_Knight =>  MVV_LVA_Table (Captured, Capturing) := 24;
		     when White_Bishop | Black_Bishop => MVV_LVA_Table (Captured, Capturing) := 34;
		     when White_Rook | Black_Rook => MVV_LVA_Table (Captured, Capturing) := 44;
		     when White_Queen | Black_Queen => MVV_LVA_Table (Captured, Capturing) := 54;
		     when White_King | Black_King => MVV_LVA_Table (Captured, Capturing) := 64;
		     when others => MVV_LVA_Table (Captured, Capturing) := 0;
		  end case;
	       when White_Queen | Black_Queen =>
		  case Captured is
		     when White_Pawn | Black_Pawn => MVV_LVA_Table (Captured, Capturing) := 15;
		     when White_Knight | Black_Knight =>  MVV_LVA_Table (Captured, Capturing) := 25;
		     when White_Bishop | Black_Bishop => MVV_LVA_Table (Captured, Capturing) := 35;
		     when White_Rook | Black_Rook => MVV_LVA_Table (Captured, Capturing) := 45;
		     when White_Queen | Black_Queen => MVV_LVA_Table (Captured, Capturing) := 55;
		     when White_King | Black_King => MVV_LVA_Table (Captured, Capturing) := 65;
		     when others => MVV_LVA_Table (Captured, Capturing) := 0;
		  end case;
	       when others => MVV_LVA_Table (Captured, Capturing) := 0;
	    end case;
	 end loop;
      end loop;

      -- init pinning piece table
      -- p1 is the candidate pinned piece
      -- p2 is the x-ray piece
      for P1 in Piece_Type'Range loop
	 for P2 in Piece_Type'Range loop
	    case P1 is
	       when White_Pawn | Black_Pawn =>
		  case P2 is
		     when White_Knight | Black_Knight | White_Bishop | Black_Bishop | White_Rook | Black_Rook | White_Queen | Black_Queen | White_King | Black_King => Pinning_Piece_Table (P1, P2) := True;
		     when others => Pinning_Piece_Table (P1, P2) := False;
		  end case;
	       when White_Knight | Black_Knight =>
		  case P2 is
		     when White_Pawn | Black_Pawn | White_Knight | Black_Knight | White_Bishop | Black_Bishop  => Pinning_Piece_Table (P1, P2) := False;
		     when White_Rook | Black_Rook | White_Queen | Black_Queen | White_King | Black_King => Pinning_Piece_Table (P1, P2) := True;
		     when others => Pinning_Piece_Table (P1, P2) := False;
		  end case;
	       when White_Bishop | Black_Bishop =>
		  case P2 is
		     when White_Pawn | Black_Pawn | White_Knight | Black_Knight | White_Bishop | Black_Bishop  => Pinning_Piece_Table (P1, P2) := False;
		     when White_Rook | Black_Rook | White_Queen | Black_Queen | White_King | Black_King => Pinning_Piece_Table (P1, P2) := True;
		     when others => Pinning_Piece_Table (P1, P2) := False;
		  end case;
	       when White_Rook | Black_Rook =>
		  case P2 is
		     when White_Pawn | Black_Pawn | White_Knight | Black_Knight |  White_Bishop | Black_Bishop | White_Rook | Black_Rook => Pinning_Piece_Table (P1, P2) := False;
		     when White_Queen | Black_Queen | White_King | Black_King => Pinning_Piece_Table (P1, P2) := True;
		     when others => Pinning_Piece_Table (P1, P2) := False;
		  end case;
	       when White_Queen | Black_Queen =>
		  case P2 is
		     when White_King | Black_King => Pinning_Piece_Table (P1, P2) := True;
		     when others => Pinning_Piece_Table (P1, P2) := False;
		  end case;
	       when others => Pinning_Piece_Table (P1, P2) := False;
	    end case;
	 end loop;
      end loop;

--        -- generate forking table
--        -- p1 is the attacking piece
--        -- p2 is the attacked piece
--        for P1 in Piece_Type'Range loop
--  	 for P2 in Piece_Type'Range loop
--  	    case P1 is
--  	       when White_Pawn | Black_Pawn =>
--  		  case P2 is
--  		     when White_Knight | Black_Knight | White_Bishop | Black_Bishop | White_Rook | Black_Rook | White_Queen | Black_Queen | White_King | Black_King =>
--  			Forking_Piece_Table (P1, P2) := True;
--  		     when White_Pawn | Black_Pawn => Forking_Piece_Table (P1, P2) := False;
--  		     when others => Forking_Piece_Table (P1, P2) := False;
--  		  end case;
--  	       when White_Knight | Black_Knight =>
--  		  case P2 is
--  		     when White_Pawn | Black_Pawn | White_Knight | Black_Knight => Forking_Piece_Table (P1, P2) := False;
--  		     when White_Bishop | Black_Bishop | White_Rook | Black_Rook | White_Queen | Black_Queen | White_King | Black_King =>
--  			Forking_Piece_Table (P1, P2) := True;
--  		     when others => Forking_Piece_Table (P1, P2) := False;
--  		  end case;
--  	       when White_Bishop | Black_Bishop =>
--  		  case P2 is
--  		     when White_Pawn | Black_Pawn | White_Bishop | Black_Bishop | White_Queen | Black_Queen =>
--  			Forking_Piece_Table (P1, P2) := False;
--  		     when White_Knight | Black_Knight | White_Rook | Black_Rook | White_King | Black_King =>
--  			Forking_Piece_Table (P1, P2) := True;
--  		     when others => Forking_Piece_Table (P1, P2) := False;
--  		  end case;
--  	       when White_Rook | Black_Rook =>
--  		  case P2 is
--  		     when White_Pawn | Black_Pawn | White_Rook | Black_Rook | White_Queen | Black_Queen =>
--  			Forking_Piece_Table (P1, P2) := False;
--  		     when White_Bishop | Black_Bishop | White_Knight | Black_Knight | White_King | Black_King =>
--  			Forking_Piece_Table (P1, P2) := True;
--  		     when others => Forking_Piece_Table (P1, P2) := False;
--  		  end case;
--  	       when White_Queen | Black_Queen =>
--  		  case P2 is
--  		     when White_Pawn | Black_Pawn | White_Bishop | Black_Bishop | White_Queen | Black_Queen =>
--  			Forking_Piece_Table (P1, P2) := False;
--  		     when White_Knight | Black_Knight | White_Rook | Black_Rook | White_King | Black_King =>
--  			Forking_Piece_Table (P1, P2) := True;
--  		     when others => Forking_Piece_Table (P1, P2) := False;
--  		  end case;
--  		  -- king should be excluded from forks due to
--  		  -- avoid rambo-king!
--  	       when others => Forking_Piece_Table (P1, P2) := False;
--  	    end case;
--  	 end loop;
--        end loop;


      for H in Ply_Type'Range loop
	 Killer_Score_3 (H) := 0;
	 Killer_Score_2 (H) := 0;
	 Killer_Score_1 (H) := 0;
	 Killer_Heuristic_3 (H) := No_Move;
	 Killer_Heuristic_2 (H) := No_Move;
	 Killer_Heuristic_1 (H) := No_Move;
      end loop;

      Generate_Moves;

   end Initialize;
   
   
   -----------------------
   -- Reset_Piece_Table --
   -----------------------  

   procedure Reset_Piece_Table is
   begin
      Piece_Table := (others => Frame);
      Piece_Table (A1 .. H1) := (others => Empty);
      Piece_Table (A2 .. H2) := (others => Empty);
      Piece_Table (A3 .. H3) := (others => Empty);
      Piece_Table (A4 .. H4) := (others => Empty);
      Piece_Table (A5 .. H5) := (others => Empty);
      Piece_Table (A6 .. H6) := (others => Empty);
      Piece_Table (A7 .. H7) := (others => Empty);
      Piece_Table (A8 .. H8) := (others => Empty);
   end Reset_Piece_Table;

   
   -----------------------
   -- Aligh_Piece_Table --
   -----------------------
   
   procedure Align_Piece_Table is
   begin
      Reset_Piece_Table;
      for I in White_Pieces'First .. White_Pieces'First + White_Pieces_Counter - 1 loop
	 Piece_Table (White_Pieces (I).Square) := I;
      end loop;
      for I in Black_Pieces'First .. Black_Pieces'First + Black_Pieces_Counter - 1 loop
	 Piece_Table (Black_Pieces (I).Square) := I;
      end loop;
   end Align_Piece_Table;


   -----------
   -- Reset --
   -----------
   
   procedure Reset (Flag : out Flag_Type) is
   begin
      Flag := No_Flags;
   end Reset;


   -------------------------
   -- Change_Side_To_Move --
   -------------------------
   
   procedure Change_Side_To_Move is
   begin
      Opponent := Side_To_Move;
      if Side_To_Move = White then
	 Side_To_Move := Black;
      else
	 Side_To_Move := White;
      end if;
   end Change_Side_To_Move;


   ------------------------
   -- Update_White_Piece --
   ------------------------

   procedure Update_White_Piece (From, To, Piece : in Integer) is
      Index : Integer;
   begin
      Index := Lookup_White_Piece (From);
      White_Pieces (Index) := (To, Piece);
      Piece_Table (From) := Empty;
      Piece_Table (To) := Index;
   end Update_White_Piece;


   ------------------------
   -- Update_Black_Piece --
   ------------------------

   procedure Update_Black_Piece (From, To, Piece : in Integer) is
      Index : Integer;
   begin
      Index := Lookup_Black_Piece (From);
      Black_Pieces (Index) := (To, Piece);
      Piece_Table (From) := Empty;
      Piece_Table (To) := Index;
   end Update_Black_Piece;


   ---------------------
   -- Add_White_Piece --
   ---------------------

   procedure Add_White_Piece (Square, Piece : in Integer) is
      Where_To : Integer;
   begin
      pragma Assert (White_Pieces'First + White_Pieces_Counter in White_Pieces'Range, "Impossible to add a white piece because there are 16 white piece on board!");
      Where_To := White_Pieces'First + White_Pieces_Counter;
      White_Pieces (Where_To) := (Square, Piece);
      Piece_Table (Square) := Where_To;
      White_Pieces_Counter := White_Pieces_Counter + 1;
   end Add_White_Piece;


   ---------------------
   -- Add_White_Piece --
   ---------------------

   procedure Add_Black_Piece (Square, Piece : in Integer) is
      Where_To : Integer;
   begin
      pragma Assert (Black_Pieces'First + Black_Pieces_Counter in Black_Pieces'Range, "Impossible to add a black piece because there are 16 black piece on board!");
      Where_To := Black_Pieces'First + Black_Pieces_Counter;
      Black_Pieces (Where_To) := (Square, Piece);
      Piece_Table (Square) := Where_To;
      Black_Pieces_Counter := Black_Pieces_Counter + 1;
   end Add_Black_Piece;

   
   ------------------------
   -- Delete_White_Piece --
   ------------------------

   procedure Delete_White_Piece (Square : in Integer) is
      Index : Integer;
   begin
      Index := Lookup_White_Piece (Square);
      pragma Assert (Index in White_Pieces'Range, "Error while deleting a White piece! Index not in White_Pieces'Range!");
      White_Pieces (Index) := White_Pieces (White_Pieces_Counter);
      Piece_Table (White_Pieces (Index).Square) := Index;
      White_Pieces (White_Pieces_Counter) := No_Piece;
      White_Pieces_Counter := White_Pieces_Counter - 1;
      Piece_Table (Square) := Empty;
   end Delete_White_Piece;


   ------------------------
   -- Delete_Black_Piece --
   ------------------------

   procedure Delete_Black_Piece (Square : in Integer) is
      Index : Integer;
   begin
      Index := Lookup_Black_Piece (Square);
      pragma Assert (Index in Black_Pieces'Range, "Error while deleting a Black piece! Index not in Black_Pieces'Range!");
      Black_Pieces (Index) := Black_Pieces (Black_Pieces_Counter );
      Piece_Table (Black_Pieces (Index).Square) := Index;
      Black_Pieces (Black_Pieces_Counter ) := No_Piece;
      Black_Pieces_Counter := Black_Pieces_Counter - 1;
      Piece_Table (Square) := Empty;
   end Delete_Black_Piece;


   ------------------------
   -- Lookup_White_Piece --
   ------------------------

   function Lookup_White_Piece (Square : in Integer) return Integer is -- dalla casa ritorna l'indice dell'array
   begin
      pragma Assert ((Piece_Table (Square) /= Empty), "Error while getting Piece_Table on Square " & Pc_Sqr (Square) & " for White" );
      return Piece_Table (Square);
   end Lookup_White_Piece;


   ------------------------
   -- Lookup_Black_Piece --
   ------------------------

   function Lookup_Black_Piece (Square : in Integer) return Integer is -- dalla casa ritorna l'indice dell'array
   begin
      pragma Assert ((Piece_Table (Square) /= Empty), "Error while getting Piece_Table on Square " & Pc_Sqr (Square)  & " for Black" );
      return Piece_Table (Square);
   end Lookup_Black_Piece;


   -------------
   -- Display --
   -------------
   
   procedure Display is
      Symbols : array (0 .. 13) of Character := ( ' ', ' ', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k' );
      Row     : Integer := 8;
      Piece   : Integer;
   begin

      --Display_Piece_Table;

      for I in ChessBoard'Range loop
	 Piece := ChessBoard (I);
	 if Piece /= Frame then
	    if Piece =  Empty then
	       Put ('.');
	    else
	       Put (Symbols (Piece));
	    end if;
	 end if;
	 Put (" ");
	 case I is
	    when 11 | 23 | 35 | 47 | 59 | 71 | 83 | 95 | 107 | 119 =>
	       if I in 24 .. 119 then
		  Put (Row, 0);
		  Row := Row - 1;
	       end if;
	       New_Line;
	    when others => null;
	 end case;
      end loop;
      New_Line;
      Put_Line (ASCII.CR & "  a b c d e f g h");
      New_Line;
   end Display;

   
   -------------------------
   -- Display_Piece_Table --
   -------------------------

   procedure Display_Piece_Table is
      Symbols : array (0 .. 13) of Character := ( ' ', ' ', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k' );
      Row     : Integer := 8;
      Piece   : Integer;
   begin
      for I in Piece_Table'Range loop
	 Piece := Piece_Table (I);
	 if Piece /= Frame then
	    if Piece =  Empty then
	       Put ('.');
	    else
	       Put (Piece, 0);
	    end if;
	 end if;
	 Put (" ");
	 case I is
	    when 11 | 23 | 35 | 47 | 59 | 71 | 83 | 95 | 107 | 119 =>
	       if I in 24 .. 119 then
		  Put (Row, 0);
		  Row := Row - 1;
	       end if;
	       New_Line;
	    when others => null;
	 end case;
      end loop;
      New_Line;
      Put_Line (ASCII.CR & "  a b c d e f g h");
      New_Line;
   end Display_Piece_Table;

   
   ----------------------
   -- Put_White_Pieces --
   ----------------------

   procedure Put_White_Pieces is
      Symbols : array (0 .. 13) of Character := ( ' ', ' ', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k' );
      M       : Move_Type;
   begin
      Put (White_Pieces_Counter, 0);
      Put (" white pieces:");
      New_Line;
      for I in White_Pieces'First .. White_Pieces_Counter loop
	 Put (White_Pieces (I).Piece, 0);
	 Put (" ");
	 Put (Symbols (White_Pieces (I).Piece));
	 Put (" on ");
	 M.From := White_Pieces (I).Square;
	 M.To := 0;
	 Echo (M);
	 New_Line;
      end loop;
   end Put_White_Pieces;
   
   
   ----------------------
   -- Put_Black_Pieces --
   ----------------------

   procedure Put_Black_Pieces is
      Symbols : array (0 .. 13) of Character := ( ' ', ' ', 'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k' );
      M       : Move_Type;
   begin
      Put (Black_Pieces_Counter, 0);
      Put (" black pieces:");
      New_Line;
      for I in Black_Pieces'First .. Black_Pieces_Counter loop
	 Put (Black_Pieces (I).Piece, 0);
	 Put (" ");
	 Put (Symbols (Black_Pieces (I).Piece));
	 Put (" on ");
	 M.From := Black_Pieces (I).Square;
	 M.To := 0;
	 Echo (M);
	 Put (Black_Pieces (I).Square);
	 New_Line;
      end loop;
   end Put_Black_Pieces;

   ----------------------
   -- Put_Moves_List --
   ----------------------

   procedure Put_Moves_List is
      Move : Move_Type;
   begin
      -- stampa mosse a schermo
      New_Line;
      Put ("Total: ");
      Put (Moves_Counter (Ply), 0);
      Put (" moves");
      New_Line;
      for I in Ply_Type'Range loop
	 exit when Moves_List (Ply, I) = No_Move;
	 Echo (Moves_List (Ply, I));
	 Put (" ");
      end loop;
      New_Line;
--        Put ("White king is on ");
--        Move.From := White_King_Position;
--        Move.To := 0;
--        Echo (Move);
--        New_Line;
--        Put ("Black king is on ");
--        Move.From := Black_King_Position;
--        Move.To := 0;
--        Echo (Move);
--        New_Line;
      if Is_In_Board (En_Passant) then
	 Put ("En passant on ");
	 Move.From := En_Passant;
	 Echo (Move);
	 New_Line;
      end if;

   end Put_Moves_List;


   ---------
   -- "=" --
   ---------

   function "=" (Left, Right : in Move_Type) return Boolean is
   begin
      return Left.From = Right.From and then Left.To = Right.To and then Left.Promotion = Right.Promotion;
   end "=";


end ACChessBoard;
