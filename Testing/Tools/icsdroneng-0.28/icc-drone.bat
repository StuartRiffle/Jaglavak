@ECHO OFF

SET ENGINEPATH=../Jaglavak-1.5.0
SET ENGINE=javlavak-1.5.0.exe
SET PROGRAM=./polyglot.exe -ec %ENGINE% -ed %ENGINEPATH% polyglot-javlavak.ini
SET OWNER=StuartRiffle
SET PROXYHOST=127.0.0.1

SET DRONERC=icc-login.ini
SET PGNFILE=icc-games.pgn
SET SHORTLOGFILE=icc-log.txt
SET TERMINFO=./terminfo
SET ICS=chessclub.com
set ICSPORT=5079
SET TIMESEAL=./timeseal.exe
SET GAMESTART=gamestart\ngamestart2\ngamestart3\ngamestart4
SET GAMEEND=gameend\ngameend2\ngameend3\ngameend4
SET HOME=.
SET VARIANTS=lightning,blitz,standard

icsdrone.exe -logging "on" -pgnLogging "on" -pgnFile "%PGNFILE%" -shortLogging "on" -shortLogFile "%SHORTLOGFILE%" -autoJoin "off" -ownerQuiet "on" -resign "off" -program "%PROGRAM%" -icsHost "%ICS%" -icsPort "%ICSPORT%" -timeseal "%TIMESEAL%" -owner "%OWNER%" -sendGameEnd "%GAMEEND%" -sendGameStart "%GAMESTART%" -loginScript "%DRONERC%" -feedback "off" -variants "%VARIANTS%" 


