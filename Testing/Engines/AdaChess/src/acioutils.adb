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
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
--with Gnat.String_Split; use Gnat.String_Split;
with ACChessBoard; use ACChessBoard;

package body ACIOUTils is


   -----------------
   -- Parse_Input --
   -----------------

   function Parse_Input (Input : in String) return Parameter_Type is
   begin
      return Split (Input);
   end Parse_Input;


   -----------------
   -- Split --
   -----------------

   function Split (Str : in String) return Parameter_Type is
      Space      : Character := ' ';
      Tokens     : Integer := 1;
      Last_Index : Integer := 0;
      Params     : Parameter_Type;
      Move       : Move_Type;
   begin
      -- conta i tokens
      for I in Str'Range loop
	 if Str (I) = Space or else I = Str'Last then
	    if Tokens = 1 then
	       if I = Str'Last then
		  Params.Command := To_Unbounded_String (Str (Str'First .. I ));
	       else
		  Params.Command := To_Unbounded_String (Str (Str'First .. I - 1));
	       end if;
	    else
	       if I = Str'Last then
		  Params.Params (Params.Params'First + Tokens - 2) := To_Unbounded_String (Str (Last_Index + 1 .. I));
	       else
		  Params.Params (Params.Params'First + Tokens - 2) := To_Unbounded_String (Str (Last_Index + 1 .. I - 1));
	       end if;
	    end if;
	    Last_Index := I;
	    Tokens := Tokens + 1;
	 end if;
      end loop;
      Tokens := Tokens - 1;
      if Tokens = 1 then
	 -- try if this is a move
	 if To_String (Params.Command)'Length in 4 .. 5 and then -- can be something like e2e4 or d7d8q
	   To_String (Params.Command) (To_String (Params.Command)'First) in 'a' .. 'h' and then
	   To_String (Params.Command) (To_String (Params.Command)'First + 1) in '1' .. '8' and then
	   To_String (Params.Command) (To_String (Params.Command)'First + 2) in 'a' .. 'h' and then
	   To_String (Params.Command) (To_String (Params.Command)'First + 3) in '1' .. '8' then
	    Move := Parse_Input_Move (To_String (Params.Command));
	    if Move /= No_Move then
	       Params.Params (Params.Params'First) := Params.Command;
	       Params.Command := To_Unbounded_String ("move");
	    end if;
	 end if;
      end if;

      Params.Tokens := Tokens;
      --        Put ("There are ");
      --        Put (Tokens, 0);
      --        Put ( " tokens:");
      --        New_Line;
      --        Put (To_String (Params.Command));
      --        Put (Length (Params.Command) );
      --        New_Line;
      --        for I in 1 .. Tokens - 2 loop
      --  	 Put (To_String (Params.Params (Params.Params'First + I - 1)));
      --  	 Put (Length (Params.Params (Params.Params'First + I - 1)));
      --  	 New_Line;
      --        end loop;
      return Params;
   end Split;


end ACIOUtils;
