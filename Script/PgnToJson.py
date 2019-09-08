import sys
import json
import chess
import chess.pgn
import zipfile
import random
import glob
import argparse


parser = argparse.ArgumentParser( description = 'PGN to JSON converter', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( 'filespec', help = 'the .pgn files to process' )
parser.add_argument( 'prefix', help = 'name prefix for the generated .pgn.json.zip' )
parser.add_argument( '--games-per-file', metavar = 'N', type = int, help = 'number of games per output split',  default = 100000 )
parser.add_argument( '--include-draws', action="store_true", help = 'export draw games too', default = False )
args = parser.parse_args()


pgnfilespec = sys.argv[1]
prefix = sys.argv[2]
games_per_file = 100000

biglist = []
rec_idx = 0
file_idx = 0
rec_this_file = 0
include_draws = True
discarded = 0
processed = 0

for pgnfile in glob.glob(args.filespec):
    pgn = open(pgnfile, encoding='utf-8-sig')

    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except:
            print( "retry" )
            continue
        if game == None:
            break

        if not include_draws:
            if game.headers['Result'] == '1/2-1/2':
                continue

        try:
            uci_moves = ''
            for move in game.mainline_moves():
                uci = chess.Move.uci( move )
                uci_moves = uci_moves + ' ' + uci
        except:
            discarded = discarded + 1
            continue

        rec = {}
        rec['result'] = game.headers['Result']
        rec['moves'] = uci_moves.strip()
        biglist.append(rec)

        processed = processed + 1
        if int( processed ) % 100 == 0:
            print( processed )

print( len(biglist), "games loaded", discarded, "discarded" )
random.shuffle( biglist )
print( "...and shuffled!" )

while len( biglist ) > 0:
    chunk = biglist[:args.games_per_file]
    biglist = biglist[len(chunk):]

    filename = args.prefix + '-' + str(file_idx).zfill( 2 ) + '.pgn.json'
    file_idx = file_idx + 1

    with zipfile.ZipFile( filename + '.zip', 'w', zipfile.ZIP_DEFLATED ) as z:
        with z.open(filename, 'w') as jsonfile:
            json_str = json.dumps(chunk, indent=4)
            jsonfile.write(json_str.encode('utf-8'))

