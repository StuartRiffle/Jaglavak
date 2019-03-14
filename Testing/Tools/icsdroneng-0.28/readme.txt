Quickstart: edit then run bot.bat

Additional info on icsdrone proxy is in README.

This update uses an old version of CYGWIN so that a
single package can support every Windows version.
There doesn't appear to be any significant downside
to this decision, but if you encounter one, let me
know on FICS or talkchess.

To operate icsdrone, you need to have 3 text files (2
for WB engines) working together.  Their filenames don't
matter, but they do reference each other.

Edit BOT.BAT
------------
  Adjust OWNER, PROGRAM, ENGINE, ENGINEPATH so that
  icsdrone can find the chess program.  Xboard engines
  will not need ENGINE or ENGINEDIR, but will require
  a siginificantly different PROGRAM setting.  You
  might also want to increase security by changing
  PROXYHOST.  See README for how that works.
  
  UCI engines might want to modify bot.ini.
  
  Adjust DRONERC so that the bot will login correctly
  to the server.  In this example, it's called
  BOT-LOGIN.INI.
  
  The sample BOT.BAT expects use of a polyglot opening
  book.  I have included fruitbook.bin which is used
  by default.  If your engine uses its own book, then
  remove the -book "%BOOK%" argument from the .BAT file.
  

The default configuration uses an alias on the server
called 'kibitzalias' for feedback.  By using this method,
you can instruct the bot to whisper, kibitz, tell
you, or other communication to output engine thinking.

Edit BOT-LOGIN.INI
------------------
If you have a registered engine, then most of the contents
of this file are not needed-- you only need to provide a
name and password for registered engines.  You set up the
aliases once, and then don't create them each login.  This
example login script is for unregistered engines.

  * adjust the kibitzalias for feedback.  use 'whisper $@'
    to make it whisper, 'kibitz $@' for kibitz.  The options
    are unlimited, but remember the $@ sequence is needed.
    You could 'tell <someplayer> $@' too.  Unalias it to
    silence the output.
  * adjust the rest of the options as you see fit.
  * the example includes "set tzone <zone>".  icsdrone
    tries to set the timezone itself, but it may not use
    a string recognized by the server.  If you want to use
    a server recognized string, add a set tzone line to
    accomplish that.  Even registered accounts will probably
    want to do this.

Once the three files are customized to your needs, just
double-click the .BAT file to run.  Again, see overview.txt
for a guide to icsdrone operation.

Once your engine connects properly and plays, you might want
to use the proxy feature.  Icsdrone should have opened a port
5000 on the local host that winboard can now connect to, eg:

  winboard.exe -ics -icshost 127.0.0.1


Source code for icsdrone is available at:
  http://alpha.uhasselt.be/Research/Algebra/Toga/icsdroneng-release/

If it is unavailable for some reason, contact nematocyst on FICS
(freechess.org), and I will get you a copy.
