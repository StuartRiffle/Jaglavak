import sys
import io
import json
import chess
import chess.engine

engineName = sys.argv[1]
jsonfile = sys.argv[2]

all_epd = []
movetime = 10

with open(jsonfile) as epd:
    teststr = epd.read()

tests = json.loads( teststr )
numdone = 0
numtotal = len( tests )
numcorrect = 0
engine = chess.engine.SimpleEngine.popen_uci( engineName )

for pos in tests:
    board = chess.Board( pos['fen'] )

    result = engine.play( board, chess.engine.Limit( time=movetime ) )

    if (str(result.move) == bmove.uci()):
       numcorrect = numcorrect + 1

    print( pos['id']+":", result.move, "(bm", bmove.uci()+")" )

print( numcorrect, "/", numdone, "correct: ", (numcorrect * 100) / numdone )

engine.quit()
