..\Tools\cutechess-cli\cutechess-cli ^
-debug ^
-engine name="Jaglavak" ^
cmd=\dev\Jaglavak\Project\x64\Release\JaglavakTest.exe ^
proto=uci ^
restart=on ^
st=10 ^
-wait 1000 ^
-engine st=1 conf="Philemon c" ^
-engine st=1 conf="Shallow Blue 2.0.0" ^
-engine st=1 conf="Apollo 1.2.1" ^
-engine st=1 conf="Sayuri 2018.05.23" ^
-engine st=1 conf="CDrill 1800" ^
-engine st=1 conf="Deepov 0.4" ^
-engine st=1 conf="Eia 0.1" ^
-engine st=1 conf="Wowl 1.3.8" ^
-engine st=1 conf="Sissa 2.0.0" ^
-openings file=..\Tools\cutechess-cli\TCEC9.pgn ^
-pgnout gauntlet.pgn ^
-tournament gauntlet ^
-each timemargin=500 restart=on ^
-games 1 ^
-repeat ^
