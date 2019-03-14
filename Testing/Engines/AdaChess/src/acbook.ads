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
with Ada.Numerics.Discrete_Random;
with ACChessBoard;	use ACChessBoard;

package ACBook is

   Book : File_Type;
   Book_Name : constant String (1 .. 13) := "adachess.book"; -- http://chessprogramming.wikispaces.com/CPW-Engine_book
   -- use only this file for book. In this version of the engine
   -- the player is forced to use "adachess.book" as book file.

   subtype Book_Opening_Lines_Range is Natural range 0 .. 61100;
   -- To avoid allocating too much memory, we will limit
   -- book entrance to this range. If you need bigger book
   -- then change this range to a larger one.

   End_Of_Book_Line : exception;

   package Random_Opening is new Ada.Numerics.Discrete_Random (Natural);
   use Random_Opening;
   -- This package will allow to pick a random move from the book

   Seed_Generator : Random_Opening.Generator;
   -- this is the seed generator for random number
   -- used to pick a random book move from
   -- the list of move founded


   procedure Open_Book;
   -- Initialize openging book "engine" by open Book file
   -- and resetting the random number generator.

   procedure Close_Book;
   -- Just close the book ;-)

   function Book_Move return Move_Type;
   -- Parse book file and find all available moves
   -- Then pick a random one.

end ACBook;
