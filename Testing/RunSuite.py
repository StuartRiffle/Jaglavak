import sys
import io
import json
import chess
import chess.engine

engineName = sys.argv[1]
jsonfile = sys.argv[2]

all_epd = []
movetime = 30

with open(jsonfile) as epd:
    teststr = epd.read()

tests = json.loads( teststr )
numdone = 0
numtotal = len( tests )
numcorrect = 0

for pos in tests:
    board = chess.Board( pos['fen'] )

    bm = pos['bm']
    try:
        bmove = chess.Move.from_uci( bm )
    except:
        bmove = board.parse_san( bm )

    engine = chess.engine.SimpleEngine.popen_uci( engineName )
    result = engine.play( board, chess.engine.Limit( time=movetime ) )
    engine.quit()

    if (str(result.move) == bmove.uci()):
       numcorrect = numcorrect + 1

    print( pos['id']+":", result.move, "(bm", bmove.uci()+")" )

print( numcorrect, "/", numdone, "correct: ", (numcorrect * 100) / numdone )
