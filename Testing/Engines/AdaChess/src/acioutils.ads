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



with Ada.Strings.Unbounded;
use Ada.Strings.Unbounded;


package ACIOUtils is

   type String_Access is access String;
   type Parameters_Array is array (0 .. 9) of Unbounded_String;

   type Parameter_Type is
      record
	 Command : Unbounded_String;
	 Params  : Parameters_Array;
	 Tokens  : Integer;
      end record;

   function Parse_Input (Input : in String) return Parameter_Type;

private

   function Split (Str : in String) return Parameter_Type;


end ACIOUtils;
