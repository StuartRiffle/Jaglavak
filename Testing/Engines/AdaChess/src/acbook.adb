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



with Ada.Directories;
with ACChessBoard;	use ACChessBoard;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Text_IO;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;
with Interfaces; use Interfaces;


package body ACBook is

   ---------------
   -- Open_Book --
   ---------------

   procedure Open_Book is
   begin
      Reset (Seed_Generator);
      if Ada.Directories.Exists (Book_Name) then
         Ada.Text_IO.Open (File => Book, Mode => In_File, Name => Book_Name);
      else
         Put_Line ("Can't find The Sacred Book file: " & Book_Name);
      end if;
   end Open_Book;


   ----------------
   -- Close_Book --
   ----------------

   procedure Close_Book is
   begin
      if Ada.Text_IO.Is_Open (Book) then
         Ada.Text_IO.Close (Book);
      end if;
   end Close_Book;


   ----------------
   -- Book_Move --
   ----------------

   function Book_Move return Move_Type is
      Moves_Line    : Unbounded_String;
      Line          : String (1 .. 512);
      Line_Length   : Natural;
      Opening_Moves : array (Book_Opening_Lines_Range) of String (1 .. 4); -- allowed only 61100 moves.
      Counter       : Natural;
      Moves         : Integer;
      Book_Line_Parsed : Natural := 0;
      Choose : Natural; -- Random move from the list of moves found in the book
      Book_Reading_Exception : exception;
   begin

      -- don't try to open book if it is closed.
      -- if book file is closed, then it means that
      -- book should not be used.
      if not Ada.Text_IO.Is_Open (Book) then
         return No_Move;
      end if;

      Moves_Line := To_Unbounded_String ("");
      Line := (others => ASCII.CR);
      Line_Length := 0;
      for I in Opening_Moves'Range loop
         Opening_Moves (I) := (others => ' ');
      end loop;

      Counter := 0;
      Moves := 0;
      Choose := 0;

      for I in History_Moves'Range loop
	 exit when I >= Ply;
	 -- echo2 return moves like e2e4e7e5d2d3
	 -- we need to append a space any time to have a
	 -- string like "e2e4 e7e5 d2d3 " and then after all
	 -- we need to remove the trailing space
	 Moves_Line := Moves_Line & Echo2 (History_Moves (I)) & " ";
      end loop;
      Moves_Line := Ada.Strings.Unbounded.Trim (Moves_Line, Side => Ada.Strings.Right);
--        Put_Line (To_String (Moves_Line) & "|");

      Ada.Text_IO.Reset (Book, In_File); -- go to the beginning of the opening book
      while not Ada.Text_IO.End_Of_File (Book) loop
	 exit when Counter >= Opening_Moves'Last - 1;
	 Ada.Text_IO.Get_Line (Book, Line, Line_Length);
         -- at the first pass, every line will match with 0-length string. That's ok:
         -- we want the first move to be picked up randomly from the book
	 if To_String (Moves_Line) (1 .. Length (Moves_Line) - 1) = Line (1 .. Length (Moves_Line) - 1)
	 or else To_String (Moves_Line) (1 .. Length (Moves_Line) - 1) = Line (1 .. Length (Moves_Line)) then
	    Opening_Moves (Counter) := To_String (Trim (To_Unbounded_String (Line (Length (Moves_Line) + 1 .. Length (Moves_Line) + 5)), Ada.Strings.Both));
	    Counter := Counter + 1;
--  	    Put_Line ("Found " &  Opening_Moves (Counter));
	 end if;
	 Book_Line_Parsed := Book_Line_Parsed + 1;
      end loop;

      if Counter > 0 then
	 -- Great! We have found one or more book moves!
         -- Now pick a random one - good luck!
         Choose := Random (Seed_Generator) rem Counter;
--           Put ("The Sacred Book of Opening Moves say: ");
--           Put_Line (Opening_Moves (Choose));
         return Parse_Input_Move (Opening_Moves (Choose));
      end if;

      return No_Move;

   exception
      when Book_Reading_Exception : End_Of_Book_Line =>
         return No_Move;
      when Book_Reading_Exception : others =>
         return No_Move;

   end Book_Move;


end ACBook;
