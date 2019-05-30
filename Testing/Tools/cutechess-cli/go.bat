cutechess-cli ^
    -debug ^
    -openings file=TCEC9.pgn ^
    ^
    -engine name=Jaglavak ^
    cmd=Jaglavak.exe ^
    dir=C:\dev\Jaglavak\Project\x64\Release ^
    proto=uci ^
    st=1000 ^
    ^
    -engine name=Stockfish ^
    cmd=stockfish_10_x64.exe ^
    dir=. ^
    proto=uci ^
    depth=9 ^
    st=1 

