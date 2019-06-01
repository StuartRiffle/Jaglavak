..\Tools\cutechess-cli\cutechess-cli ^
    -debug ^
    -tournament gauntlet ^
    ^
    -engine name="Jaglavak" ^
    cmd=\dev\Jaglavak\Project\x64\Release\Jaglavak.exe ^
    proto=uci ^
    st=10 ^
    ^
    -engine conf="Stockfish 3" ^
    timemargin=200 ^
    st=1
    ^
    -openings file=..\Tools\cutechess-cli\TCEC9.pgn ^
    -games 2 ^

 

