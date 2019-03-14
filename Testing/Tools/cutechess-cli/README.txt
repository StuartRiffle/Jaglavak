CUTECHESS-CLI(6)
================

NAME
----
cutechess-cli - A command-line tool for chess engines matches


SYNOPSIS
--------
*cutechess-cli* -engine ['ENGINE OPTIONS'] -engine ['ENGINE OPTIONS']... ['OPTIONS']

DESCRIPTION
-----------
Runs chess matches from the command line.


OPTIONS
-------

\--version::
	Display the version information.

\--help::
	Display help information.

\--engines::
	Display a list of configured engines and exit.

\--protocols::
	Display a list of supported chess protocols and exit.

\--variants::
	Display a list of supported chess variants and exit.

-engine <options>::
	Add an engine defined by <options> to the tournament.

-each <options>::
	Apply <options> to each engine in the tournament.

-variant <arg>::
	Set chess variant to <arg>.

-concurrency <n>::
	Set the maximum number of concurrent games to <n>.

-draw <n> <score>::
	Adjudicate the game as a draw if the score of both engines is
	within <score> centipawns from zero after <n> full moves have
	been played.

-resign <n> <score>::
	Adjudicate the game as a loss if an engine's score is at least
	<score> centipawns below zero for at least <n> consecutive moves.

-tournament <arg>::
	Set the tournament type to <arg>. Supported types are:
	round-robin (default)
	gauntlet

-event <arg>::
	Set the event name to <arg>.

-games <n>::
	Play <n> games per encounter. This value should be set to an even
	number in tournaments with more than two players to make sure
	that each player plays an equal number of games with white and
	black pieces.

-rounds <n>::
	Multiply the number of rounds to play by <n>. For two-player
	tournaments this option should be used to set the total number of
	games to play.

-ratinginterval <n>::
	Set the interval for printing the ratings to <n> games.

-debug::
	Display all engine input and output.

-pgnin <file>::
	Use <file> as the opening book in PGN format.

-pgndepth <n>::
	Set the maximum depth for PGN input to <n> plies.

-pgnout <file> [min]::
	Save the games to <file> in PGN format. Use the 'min' argument
	to save in a minimal PGN format.

-recover::
	Restart crashed engines instead of stopping the match.

-repeat::
	Play each opening twice so that both players get to play it on
	both sides.

-site <arg>::
	Set the site / location to <arg>.

-srand <n>::
	Set the random seed for the book move selector to <n>.

-wait <n>::
	Wait <n> milliseconds between games. The default is 0.


ENGINE OPTIONS
--------------

conf=<arg>::
	Use an engine with the name <arg> from Cute Chess\' configuration
	file.

name=<arg>::
	Set the name to <arg>.

cmd=<arg>::
	Set the command to <arg>.

dir=<arg>::
	Set the working directory to <arg>.

arg=<arg>::
	Pass <arg> to the engine as a command line argument.

initstr=<arg>::
	Send <arg> to the engine's standard input at startup.

restart=<arg>::
	Set the restart mode to <arg> which can be:
	'auto': the engine decides whether to restart (default)
	'on': the engine is always restarted between games
	'off': the engine is never restarted between games

proto=<arg>::
	Set the chess protocol to <arg>.

tc=<arg>::
	Set the time control to <arg>. The format is
	moves/time+increment, where 'moves' is the number of
	moves per tc, 'time' is time per tc (either seconds or
	minutes:seconds), and 'increment' is time increment
	per move in seconds.
	Infinite time control can be set with 'tc=inf'.

st=<n>::
	Set the time limit for each move to <n> seconds.
	This option can't be used in combination with "tc".

timemargin=<n>::
	Let engines go <n> milliseconds over the time limit.

book=<file>::
	Use <file> (Polyglot book file) as the opening book.

bookdepth=<n>::
	Set the maximum book depth (in fullmoves) to <n>.

whitepov::
	Invert the engine's scores when it plays black. This
	option should be used with engines that always report
	scores from white's perspective.

depth=<n>::
	Set the search depth limit to <n> plies.

nodes=<n>::
	Set the node count limit to <n> nodes.

option.<name>=<arg>::
	Set custom engine option <name> to value <arg>.


EXAMPLES
--------

* Play ten games between two Sloppy engines with a time
control of 40 moves in 60 seconds.

-----------
$ cutechess-cli -engine name=Sloppy -engine name=Sloppy -each cmd=sloppy proto=xboard tc=40/60 -rounds 10
-----------

* Use the 'name=Atak' parameter because it's a Xboard
protocol 1 engine and doesn't tell its name.

* Use the 'dir=C:\atak' parameter to point the location of
the executable.

* Glaurung can tell its name and is in the PATH variable
so only the command is needed.

* Set Glaurung to use 1 thread.

* Set the time control to 40 moves in one minute and 30
seconds with a one second increment.

-----------
$ cutechess-cli -engine name=Atak cmd=Atak32.exe dir=C:\atak proto=xboard -engine cmd=glaurung proto=uci option.Threads=1 -both tc=40/1:30+1
-----------

* Play a Round-Robin tournament between Fruit, Crafty, Stockfish and Sloppy.

* Play two games per encounter, effectively multiplying the number of games by 2.

* Play 10 times the minimum amount of rounds (3). So the total number of rounds
to play will be 30, and the total number of games 120.

* In each two-game encounter colors are switched between games and the same
opening line is played in both games.

-----------
$ cutechess-cli -engine conf=Fruit -engine conf=Crafty -engine conf=Stockfish -engine conf=Sloppy -each tc=4 book=book.bin -games 2 -rounds 10 -repeat -tournament round-robin
-----------

AUTHOR
------
Written by Ilari Pihlajisto <ilari.pihlajisto@mbnet.fi> and Arto Jonsson
<ajonsson@kapsi.fi>.


RESOURCES
---------
* Source code: <http://repo.or.cz/w/sloppygui.git>

* Mailing list: <https://list.kapsi.fi/listinfo/cutechess>

COPYING
-------
Copyright \(C) 2008-2012 Ilari Pihlajisto and Arto Jonsson. Free use of this
software is granted under the terms of GNU General Public License (GPL).

