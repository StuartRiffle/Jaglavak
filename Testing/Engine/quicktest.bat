..\Tools\cutechess-cli\cutechess-cli ^
    -debug ^
    -tournament gauntlet ^
    ^
    -engine name="Jaglavak" ^
    cmd=\dev\Jaglavak\Project\x64\Release\Jaglavak.exe ^
    proto=uci ^
    st=60 ^
    ^
    -engine conf="Shallow Blue 2.0.0" ^
    timemargin=200 ^
    st=1 ^
    ^
    -openings file=..\Tools\cutechess-cli\TCEC9.pgn ^
    -rounds 5


 

