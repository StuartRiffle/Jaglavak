import sys
import io
import json
import chess
import chess.engine
import logging
logging.basicConfig(level=logging.DEBUG)

engineName = sys.argv[1]
jsonfile = sys.argv[2]
movetime = int( sys.argv[3] )

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

    msg = ""
    if (str(result.move) == bmove.uci()):
        msg = " CORRECT!"
        numcorrect = numcorrect + 1

    numdone = numdone + 1

    print( pos['id']+":", pos['fen'], " (bm", bmove.uci()+")... ", result.move, msg )
    print( numcorrect, '/', numdone, '-', numcorrect/numdone )


