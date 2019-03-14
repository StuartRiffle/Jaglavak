@ECHO OFF

REM Things you should change
SET ENGINEPATH=../Corvid-1.5.1
SET ENGINE=corvid-1.5.1.exe
SET PROGRAM=./polyglot.exe -ec %ENGINE% -ed %ENGINEPATH% polyglot-corvid.ini
SET OWNER=StuartRiffle
SET PROXYHOST=127.0.0.1

SET DRONERC=fics-login.ini
SET PGNFILE=corvid-on-fics.pgn
SET SHORTLOGFILE=fics-log.txt
SET TERMINFO=./terminfo
SET ICS=freechess.org
SET TIMESEAL=./timeseal.exe
SET GAMESTART=gamestart\ngamestart2\ngamestart3\ngamestart4
SET GAMEEND=gameend\ngameend2\ngameend3\ngameend4
SET HOME=.
SET VARIANTS=lightning,blitz,standard

echo.
echo ---------------------------------------------------------------
echo REMINDER: type "daemonize" to push icsdrone into the background
echo ---------------------------------------------------------------
icsdrone.exe -logging "off" -pgnLogging "on" -pgnFile "%PGNFILE%" -shortLogging "on" -shortLogFile "%SHORTLOGFILE%" -autoJoin "off" -ownerQuiet "on" -resign "off" -program "%PROGRAM%" -icsHost "%ICS%" -timeseal "%TIMESEAL%" -owner "%OWNER%" -sendGameEnd "%GAMEEND%" -sendGameStart "%GAMESTART%" -loginScript "%DRONERC%" -feedback "off" -variants "%VARIANTS%"

