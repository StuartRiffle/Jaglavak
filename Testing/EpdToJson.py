import sys
import json
import chess
import chess.engine

epdfile = sys.argv[1]
all_epd = []

refengine = chess.engine.SimpleEngine.popen_uci("..\Engine\stockfish-10")

line_idx = 0
with open(epdfile + ".epd") as epd:
    for line in epd.readlines():
        line_idx = line_idx + 1
        line = line.strip()

        # Some epd files use 'am' instead of 'bm'
        line = line.replace( "am ", "bm " )

        # Some epd files have no separator between the fen and best move
        line = line.replace( "bm ", ";bm " ) 

        # A small number of epd files don't actually provide
        # a best move, which seems like it kind of defeats the point,
        # but fine. In these cases we fire up a strong reference engine 
        # to get a quick opinion on the position. Deeper searches might
        # give us better data here.
        if not 'bm ' in line:
            board = chess.Board( line )
            result = refengine.play( board, chess.engine.Limit( depth=10 ) )
            line = line + ";bm " + str(result.move)

        # After the fen it's all key/value pairs.
        fields = line.split( ';' )
        if len( fields ) > 0:
            this_test = {}
            fen = fields[0].strip()
            this_test['fen'] = fen

            for meta in fields[1:]:
                meta = meta.strip()
                if len( meta ) > 0:
                    if ' ' in meta:
                        sep = meta.index( ' ' )
                        key = meta[:sep].strip()
                        val = meta[sep:].strip()

                        if val.startswith( '"' ) and val.endswith( '"' ):
                            val = val[1:-1]

                        this_test[key] = val

            if not 'id' in this_test:
                this_test['id'] = epdfile.replace( '.', '-' ) + "-" + str( line_idx )

            try:
                bmove = chess.Move.from_uci( bm )
            except:
                # Oh, gross
                bmove = board.parse_san( bm )                

            all_epd.append( this_test )

ser = json.dumps( all_epd, sort_keys=True, indent=4 )
print( ser )

refengine.quit()
