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


--------------------------------------
-- Roadmap from Version 0.0 to 1.0: --
--------------------------------------

-- AdaChess v0.1
-- Hello World. Environment preparation and
-- ChessBoard description as array of 64 squares.

-- AdaChess v0.2
-- Basic set-up of chessboard, pseudo-moves piece generator

-- AdaChess v0.3
-- Play and Undo function. First primitive Evaluation function
-- helps for debug.

-- AdaChess v0.4
-- Added AlphaBeta searching algorithm

-- AdaChess v0.5
-- Implemented Sorting algorithm for mvoes.
-- Added HistoryHeuristic to help AlphaBeta doing more
-- cut offs. Added also a MVV-LVA score to capture moves.

-- AdaChess v0.6
-- Implemented the (Triangular) Principal Variation line
-- Move searching changed using iterative deepeing.

-- Adachess v0.7
-- Added Quiescence after AlphaBeta searching

-- AdaChess v0.8
-- Move generator is completed with Castle, en-passant and
-- pawn promotion. Additionally, a first incomplete FEN
-- parser is developed

-- AdaChess v0.9
-- Implemented Zobrist hashing algorithm.
-- Implemented the fifty move rules, draw by three-fold repetitions
-- Implemented first, rudimental, time-support (only on 40 moves in 5 minutes)

-- AdaChess v1.0
-- Added a first, weak, Winboard protocol support

-- Benchmark for v1.0 are estimated as
-- 10000 NPS (Nodes per Second)


--------------------------------------
-- Roadmap from Version 1.0 to 2.0: --
--------------------------------------

-- AdaChess v1.1:
-- ChessBoard Implementation changed from array(1..64) to array(1..144)
-- All the code related to the ChessBoard has been rewritten according to
-- this new implementation. The 144 board is easier for maintenance.

-- AdaChess v1.2:
-- Piece implementation changed. Piece are now identified as integers and
-- differ from white/black. The new implementation involves constants to
-- recognize out-of-board squares and empty sqares.

-- AdaChess v1.3:
-- Some utility functions has been added. Those utilities helps to detect
-- which color stands on a certain square, if a square is empty and so on.
-- the complete list is:
-- function Is_White (Piece : in Integer) return Boolean with Inline => True;
-- function Is_Black (Piece : in Integer) return Boolean with Inline => True;
-- function Is_In_Board (Square : in Integer) return Boolean with Inline => True;
-- function Is_Piece (Piece : in Integer) return Boolean with Inline => True;
-- function Is_Empty (Square  : in Integer) return Boolean with Inline => True;
-- function Is_Capture (Move : in Move_Type) return Boolean with Inline => True;

-- AdaChess v1.4:
-- Added an array of pieces to loop through (for white and black). No more
-- time spent to loop through the whole ChessBoard, now just loop through
-- those array. Special function to Add, Delete and Lookup piece here help
-- to keep update all data.

-- AdaChess v1.5:
-- Thinking function improved with new features. Aspiration window added to the
-- iterative deepening loop and a small transposition table based on hash
-- to store some previous score calculation.
-- Added a Zero-Window-Search for some moves
-- The table for evaluation function has been removed.

-- AdaChess v1.6:
-- Evaluation function has been improved with a lot of new features. Current
-- evaluation looks for and assign bonus/penalties for:
-- 0) Default: Material and piece position
-- 1) Pawn structure, isolated pawns, and so on.
-- 2) Rooks position like rooks on open file, and so on.
-- 3) King safety. Bonus for Castling and for king protection.
-- 4) While in the opening, a strong penalies for piece not developed
-- 5) Encourage capture when in advantage of material
-- 6) Added a mobility check to find blocked pawns, blocked bishops, and so on
-- 7) Pinned piece detection
-- 8) Small improvements on positional play with weak square recognition
-- 9) Unprotected piece recognition

-- AdaChess v1.7:
-- Small improvements added to the code for find File/Ranks on a given square.
-- The new code use a table-driven algorithm. Tables added for Diagonal
-- and Anti-Diagonal too.

-- AdaChess v1.8:
-- Legality test improved. Legality test is now based from the king position
-- looking reverse to find attackers. The legality test is asked only in
-- certain position. Like when a (potential) absolutely pinned piece moves
-- but not along the same path as it's (potential) attacker.
-- However, the same idea has been applied to the Find_Attack function.

-- AdaChess v1.9:
-- Improved Xboard support with new commands and parameters recognition
-- Utility to parse input added. Time management improved with new features
-- and more advanced time-consumption while thinking.

-- AdaChess v2.0:
-- Perft and Divide test added.
-- Improved move sorting with Killer Moves recognition.
-- Late Move Reduction algorithm added to the search

-- Performance:
-- AdaChess 2.0 has this performance tested with initial position on a
-- Hp TouchSmart tm2 Notebook PC - Intel 64 Bit i3 CPU 1.20 GHz
-- Windows 7 Home Premium
-- please note that current version has been targeted for 32 bit CPU

-- Perft(6)
-- Depth   Nodes  Time (seconds)
-- 1          20    0.00
-- 2         400    0.00
-- 3        8902    0.01
-- 4      197281    0.23
-- 5     4865609    4.88
-- 6   119060324  123.13
-- From the initial position, benchmark with 30 seconds of thinking
-- time gives 54075 NPS - say 55000 ;-)



with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories; 	use Ada.Directories;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
with Ada.Float_Text_IO; use Ada.Float_Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with ACChessBoard; use ACChessBoard;
with ACSearch; use ACSearch;
with ACTimeManage; use ACTimeManage;
with ACIOUtils; use ACIOUtils;
with Ada.Calendar; use Ada.Calendar;
with Ada.Exceptions;
with ACBook; 	use ACBook;
with ACHash; use ACHash;
with ACFen; use ACFen;
with ACEvaluate; use ACEvaluate;



procedure AdaChess is

   procedure Signal;
   pragma Import (C, Signal, "_signal");

   procedure Prompt is
   begin
      Put ("AdaChess v2.0 => ");
   end Prompt;

   Params : Parameter_Type; -- user input

   -- Xboard
   Xboard : Boolean;
   Forcemode : Boolean;

   Engine         : Integer := Empty;
   Move           : Move_Type;
   Input          : Unbounded_String;
   Console_Input  : String (1 .. 128);
   Last           : Integer;
   Quit           : String := "quit";
   Fen_String : Ada.Strings.Unbounded.Unbounded_String;
   
   --     Log            : File_Type;
   --     Log_File_Name  : String (1 .. 7) := "log.txt";

   Thinking_Time  : Duration;

   Engine_Moves_Iteration : Integer := 0; -- each 40 moves is an iteration
   Engine_Moves_Counter   : Integer := 0;

   Tries : Natural := 0;


   Benchmark_NPS         : array (1 .. 3) of Integer;
   Benchmark_Best_NPS    : Integer;


   procedure Usage is
   begin
      Put_Line ("AdaChess v2.0");
      New_Line;
      Put_Line ("quit - Exit game");
      Put_Line ("usage - Show this menu");
      Put_Line ("xboard - start Winboard/Xboard mode. Support those commads (available also out of xboard mode):");
      Put_Line (ASCII.HT & "protover <version> - Send supported protocol version number");
      Put_Line (ASCII.HT & "new - Start a new game. Init board, counter, time management and so on");
      Put_Line (ASCII.HT & "go - Start thinking with current color");
      Put_Line (ASCII.HT & "move <move> - Read the move and play it");
      Put_Line (ASCII.HT & "level - Set game mode");
      Put_Line (ASCII.HT & "st <seconds> - Set thinking time for each move");
      Put_Line (ASCII.HT & "sd <depth> - Set thinking depth for each move");
      Put_Line (ASCII.HT & "undo - Take back last move");
      Put_Line (ASCII.HT & "display - Print board on console");
      Put_Line (ASCII.HT & "white - Set engine play black");
      Put_Line (ASCII.HT & "black - Set engine play white");
      Put_Line ("perft <depth> [loop] - Start perft from current position at given depth.");
--        Put_Line (ASCII.HT & "divide <depth> [loop] - start divide from current position at given depth.");
      Put_Line ("bench - Perform a benchmark testing");
      Put_Line ("mlist - Print legal moves list");
      New_Line;
      Put_Line ("Moves are expected as lowercase. Example of valid moves are: e2e4, c7c8q for promotion, e1g1 for castle.");
      new_line;
   end Usage;

begin

   --     if Ada.Directories.Exists (Log_File_Name) then
   --        Ada.Directories.Delete_File (Log_File_Name);
   --     end if;
   --     Ada.Text_IO.Create (Log, Append_File, Log_File_Name);

   Ouptut_Mode := ACSearch.Standard;
   Enable_PV_Move := True;
   
   Thinking_Time := 0.0;
   Set_Time (7.0);

   Initialize;
   Update_Hash;

   Engine := Empty;

   Xboard := False;
   Forcemode := False;


   Open_Book;

   while True loop

      -- close the book after the Opening has ended
      if History_Ply > 15 then
	 Close_Book;
      end if;
      

      if not Xboard then
	 Display;
      end if;



      if Side_To_Move = Engine and then Forcemode = False then

	 Thinking_Time := Get_Thinking_Time;

	 Start_Clock;
	 Move := Think (Thinking_Time);
	 Stop_Clock;

	 --  	 Put_Line (Log, "Moving " & Echo2 (Move));
	 --  	 Flush (Log);

	 if Move = No_Move  then
	    -- this code will be executed only if there are no valid
	    -- moves found. This means that the game has ended.
	    -- The code to find if someone won the game or it is a draw
	    -- is coded later (take a look 25 line after this one)
	    null;
	 else
	    --  	    Put_Line (Log, "Move /= No_Move");
	    --  	    Flush (Log);
	    if Play (Move) then
	       --  	       Put_Line (Log, "Move played");
	       --  	       Flush (Log);
	       if Xboard then
		  Put_Line ("move " & Echo2 (Move));
		  Flush;
	       else
		  Put ("Engine moves: ");
		  Echo (Move);
		  New_Line;
	       end if;
	    end if;
	 end if;

	 if not Xboard then
	    Display;
	 end if;
      end if;


      Generate_Moves;

      if Moves_Counter (Ply) = 0 then
	 if Has_King_In_Check (White) then
	    Put_Line ("Checkmate! - Black won!");
	 elsif Has_King_In_Check (Black) then
	    Put_Line ("Checkmate! - White won!");
	 else
	    Put_Line ("Stalemate!");
	 end if;
	 return;
      end if;
      if Fifty >= 100 then
	 Put_Line ("Draw by fifty moves rule");
	 return;
      elsif Count_Repetitions >= 3 then
	 if Has_King_In_Check (White) or else Has_King_In_Check (Black) then
	    Put_Line ("Draw by perpetual check (three-fold repetition)");
	 else
	    Put_Line ("Draw by three-fold repetition");
	 end if;
	 return;
      elsif Draw_By_Insufficient_Material then
	 Put_Line ("Draw by insufficient material");
	 -- dont call "return" here.
	 -- my algorithm is not so advanced and doesn't cover ALL
	 -- situation. Also, rarely it detect draws even if there
	 -- are chances to win.
	 -- see chess.stackexchange.com/questions/5028/what-is-sufficient-mating-material
	 -- However, this is a good compromise for mostly situations.
      end if;
      

      -- Ask for user input
      if not Xboard then
	 Prompt;
      end if;

      --        Put_Line (Log, "Asking input");
      --        Flush (Log);
      --        if Forcemode then
      --  	 Put_Line (Log, "Forcemode = True");
      --        else
      --  	 Put_Line (Log, "Forcemode = False");
      --        end if;
      --        if Side_To_Move = White then
      --  	 Put_Line (Log, "Side to move = White");
      --        else
      --  	 Put_Line (Log, "Side to move = Black");
      --        end if;
      --        if Engine = White then
      --  	 Put_Line (Log, "Engine = White");
      --        elsif Engine = Black then
      --  	 Put_Line (Log, "Engine = Black");
      --        else
      --  	 Put_Line (Log, "Engine = Empty");
      --        end if;
      --        Flush (Log);

      if Side_To_Move = Engine and then Forcemode = False then
	 --  	 Put_Line (Log, "Something's wrong while reading input.. Engine turn but it isn't calculating!!");
	 --dont ask input!
	 --  	 Flush;
	 --  	 Flush (Log);
	 null;
      else
	 Get_Line (Console_Input, Last);
      end if;

      --        Put_Line (Log, "Console_Input: " & Console_Input (1 .. Last));
      --        Flush (Log);

      Input := To_Unbounded_String (Console_Input (1 .. Last));

      --        Put_Line (Log,  Console_Input (1 .. Console_Input'Last));

      Params := Parse_Input (To_String (Input));


      if Console_Input (1) = ASCII.LF then
	 -- Sometimes xboard send a new line to the engine.
	 -- Just skip it...
	 null;
      elsif Console_Input (1) = '?' then
	 Forcemode := False;

      elsif To_String (Params.Command) = "usage" then
	 Usage;
	 
      elsif To_String (Params.Command) = "quit" then
	 -- ;-) Thanks for playing AdaChess
	 return;

      elsif To_String (Params.Command) = "xboard" then
	 -- Set output type to xboard, defined in acsearch.ads,
	 -- for correct printing of Principal Variation
	 Ouptut_Mode := ACSearch.Xboard; 
	 
	 Xboard := True;
	 Signal; -- call C function _signal(SIGINT, SIG_IGN)
	 New_Line;
	 Flush;


      elsif To_String (Params.Command) = "protover" then
	 Put_Line ("feature myname=" & '"' & "AdaChess - v2.0" & '"' );
	 Flush;
	 Put_Line ("feature done=" & '"' & '1' & '"');
	 Flush;

      elsif To_String (Params.Command) = "new" then
	 Initialize;
	 Engine := Black;
	 Forcemode := False;
	 Update_Hash;

      elsif To_String (Params.Command) = "go" then
	 Forcemode := False;
	 Engine := Side_To_Move;

      elsif To_String (Params.Command) = "move" then
	 Move := Parse_Input_Move (To_String (Params.Params (Params.Params'First) ));
	 --  	 if Move = No_Move then
	 --  	    Put_Line (Log, "Invalid move!");
	 --  	 else
	 if Play (Move) then
	    null;
	 end if;

      elsif To_String (Params.Command) = "level" then
	 Set_Time (Number_Of_Moves => Integer'Value (To_String (Params.Params (Params.Params'First))),
	    Total_Time      => Duration'Value (To_String (Params.Params (Params.Params'First + 1))),
	    Increment       => Duration'Value (To_String (Params.Params (Params.Params'First + 2))));


      elsif To_String (Params.Command) = "st" then
	 if Params.Tokens /= 2 then
	    Put_Line ("Please set time with: st <seconds>");
	 else
	    Set_Time (Duration'Value (To_String (Params.Params (Params.Params'First))));
	 end if;
	 
      elsif To_String (Params.Command) = "sd" then
	 if Params.Tokens /= 2 then
	    Put_Line ("Please set depth with: sd <depth>");
	 else
	    Set_Depth (Positive'Value (To_String (Params.Params (Params.Params'First))));
	 end if;

      elsif To_String (Params.Command) = "undo" then
	 if History_Ply >= Ply_Type'First then
	    Undo;
	    Engine := Empty;
	 end if;

      elsif To_String (Params.Command) = "display" then
	 Display;

      elsif To_String (Params.Command) = "white" then
	 Engine := Black;

      elsif To_String (Params.Command) = "black" then
	 Engine := White;

      elsif To_String (Params.Command) = "force" then
	 --  	 Engine := Empty;
	 Forcemode := True;

      elsif To_String (Params.Command) = "perft" then
	 --  	 Perft (5);
	 --  	 Put_Line (Integer'Image (Params.Tokens) & " parametri" );
	 --  	    for I in Params.Params'First .. Params.Params'First + Params.Tokens - 2 loop
	 --  	       Put_Line ("Parametro" & Integer'Image (I) & ": " & To_String (Params.Params (I)));
	 --  	    end loop;
	 if Params.Tokens = 3 and then To_String (Params.Params (Params.Params'First + 1)) = "loop" then
	    for I in 1 .. Integer'Value (To_String (Params.Params (Params.Params'First))) loop
	       Perft (I);
	    end loop;

	 elsif Params.Tokens  = 2 then
	    Perft (Integer'Value (To_String (Params.Params (Params.Params'First))));

	 else
	    Put_Line ("Call: perft <depth> [loop]");
	 end if;


	 --  	    for I in 1 .. 5 loop
	 --  	       Perft (I);
	 --  	    end loop;
	 New_Line;

      elsif To_String (Params.Command) = "bench" then
	 Enable_PV_Move := False; -- dont pick the move if you found it
	 Close_Book;
	 Display;
	 Put_Line ("Benchmarking current chess position");
	 Put_Line ("30 seconds evaluations, 3 steps");
	 Flush;
	 for Test in Benchmark_NPS'Range loop
	    Put_Line ("Test number" & Integer'Image (Test));
	    delay 2.0;
	    Set_Time (Seconds => 30.0);
	    Start_Clock;
	    Move := Think (Get_Thinking_Time);
	    Stop_Clock;
	    Put ("Total nodes: ");
	    Put (Item => Float (Nodes), Fore => 3, Aft => 5, Exp => 0);
	    Put (" nps " & Integer'Image (Nodes / 30));
	    New_Line;
	    Flush;
	    Benchmark_NPS (Test) := Nodes / 30;
	 end loop;
	 -- Put the best time from the 5
	 Benchmark_Best_NPS := 0;
	 for I in Benchmark_NPS'Range loop
	    if Benchmark_NPS (I) > Benchmark_Best_NPS then
	       Benchmark_Best_NPS := Benchmark_NPS (I);
	    end if;
	 end loop;
	 Put_Line ("Best performance is:" & Integer'Image (Benchmark_Best_NPS) & " Nodes per Second");
	 Set_Time (Seconds => 7.0);
	 Open_Book;
	 Enable_PV_Move := True; -- take back the configuration

      elsif To_String (Params.Command) = "divide" then
	 Divide (4);

      elsif To_String (Params.Command) = "fen" then
--  	 Fen_Load (To_String (Fen_String));
	 Fen_Init;
	 Fen_Load_Pieces (To_String (Params.Params (Params.Params'First)));
	 Fen_Load_Side_To_Move (To_String (Params.Params (Params.Params'First + 1)));
	 Fen_Load_Castle_Flags (To_String (Params.Params (Params.Params'First + 2)));
	 Fen_Load_En_Passant (To_String (Params.Params (Params.Params'First + 3)));
	 Fen_Load_Half_Move_Clock (To_String (Params.Params (Params.Params'First + 4)));
	 Fen_Load_Fullmove_Counter (To_String (Params.Params (Params.Params'First + 5)));
	 Update_Hash;
	 
      elsif To_String (Params.Command) = "mlist" then
	 Put_Moves_List;
	 new_line;
	   
	 --        elsif To_String (Params.Command) = "fen1" then
--  	 --Fen_Load ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
--  	 --Fen_Load ("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
--  
--  --  	 Fen_Load ("8/5k2/3P4/8/2n2p2/6B1/8/3K4 w - - 0 1");
--  	 Fen_Load ("1k6/5ppp/8/8/8/8/PPP5/6K1 w - - 0 1");
--  --  	 Fen_Load ("n2r2k1/p4ppp/1p1p4/1qpN2N1/3P4/8/PP4BP/1K2R3 w - - 0");
--  	 Update_Hash;
--  	 -- raise exception se le caselle esterne non sono tutte FRAME
--        elsif To_String (Params.Command) = "fen2" then
--  	 Fen_Load ("2r5/pp4bk/6Np/2qQ2P1/4p3/6P1/PP4P1/1K3R1R w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen3" then
--  	 --Fen_Load ("2R2bk1/5rr1/p3Q3/3Ppq1R/1p3p2/8/PP1B2PP/7K w - - 0 1");
--  	 Fen_Load ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen4" then
--  	 Fen_Load ("2R2bk1/5rr1/p3Q2R/3Ppq2/1p3p2/8/PP1B2PP/7K w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen5" then
--  	 Fen_Load ("r2q3k/Q1ppRR2/3P4/2P5/8/8/8/7K w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen6" then
--  	 Fen_Load ("2bqkbnr/3r1ppp/p1npp3/8/Pp2P3/2N2N2/1P2QPPP/R4RK1 w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen7" then
--  	 Fen_Load ("8/8/8/2p5/1pp5/brpp4/qpprpK1P/1nkbn3 w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen8" then
--  	 Fen_Load ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fen9" then
--  	 Fen_Load ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fena" then
--  	 Fen_Load ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fenb" then
--  	 Fen_Load ("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fenc" then
--  	 Fen_Load ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
--  	 Update_Hash;
--        elsif To_String (Params.Command) = "fend" then
--  	 Fen_Load ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fent" then
--  	 Fen_Load ("rnbqk2r/ppp2ppp/4pn2/3p4/3P1B2/P1P1P3/2P2PPP/R2QKBNR b KQkq - 0 6");
--  	 Update_Hash;
--  	 Generate_Moves;
--  	 Put_Moves_List;
--  
--        elsif To_String (Params.Command) = "fenl" then
--  	 Fen_Load ("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - -");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fenh" then
--  	 --  	 Fen_Load ("8/p1q3pp/1p3k2/1R2p3/2r5/3b1P2/R5PP/4Q1K1 w - - 0 1");
--  	 --  	 Fen_Load ("5k2/p1QB2pp/8/8/2P5/2KP4/r6P/7R w - - 0 26");
--  	 --	 Fen_Load ("2k5/1p6/8/1p5p/r7/1K4p1/6q1/8 w - - 0 1");
--  --  	 Fen_Load ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
--  	 Fen_Load ("rn1qk2r/ppp1bppp/1n6/3pN3/2P5/8/PP1NQPPP/R1B1K2R w KQkq - 0 10");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fent2" then
--  	 Fen_Load ("4r1k1/2n5/4R3/1B1P4/r7/8/8/6K1 w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fenp" then
--  	 --  	 Fen_Load ("4Q3/k1p5/8/8/1Q3PPP/8/P7/3K4 w - - 0 1");
--  	 Fen_Load ("k7/8/8/8/8/8/8/7K w - - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "leonid" then
--  	 Fen_Load ("q2k2q1/2nqn2b/1n1P1n1b/2rnr2Q/1NQ1QN1Q/3Q3B/2RQR2B/Q2K2Q1 w - - ");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fentest1" then
--  	 Fen_Load ("rn1qkb1r/pp2pppp/5n2/3p1b2/3P4/2N1P3/PP3PPP/R1BQKBNR w KQkq - 0 1");
--  	 Update_Hash;
--  
--        elsif To_String (Params.Command) = "fens" then
--  	 Fen_Load ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
--  	 Update_Hash;
--  
--  
--        elsif To_String (Params.Command) = "fens1" then
--  	 Fen_Load ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
--  	 Update_Hash;
--      
--        elsif To_String (Params.Command) = "fens2" then
--  	 Fen_Load ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
--  	 Update_Hash;
--  	 
--        elsif To_String (Params.Command) = "fenx" then
--  	 Fen_Load ("1n2k1n1/8/8/8/8/8/8/1N2K1N1 w - - 0 1");
--  	 Update_Hash;

      else
	 if not Xboard then
--  	    Put_Line ("Unknown command");
--  	    Usage;
	    null;
	 end if;
      end if;

      -- reset input
      --        Console_Input := (others => ' ');
      --        Put_Line (Log, "");

   end loop;

   Close_Book;

exception

   when E : others =>
      -- print a trace into a log file to find where the problem is;
      Put_Line (Ada.Exceptions.Exception_Information (E));
      -- See More at : Http :  /  / Www.Adacore.Com / Adaanswers / Gems / Gem - 142 - Exceptions / #sthash.zFX3TWFc.dpuf
      Put_Line ("AdaChess catched an exception, please send me an email at adachess@gmail.com and tell me what you've done.");
      Close_Book;

end AdaChess;
