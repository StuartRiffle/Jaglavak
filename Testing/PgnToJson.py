import sys
import json
import chess
import chess.pgn
import zipfile

pgnfile = sys.argv[1]
prefix = sys.argv[2]
games_per_file = 10000

all_pgn = []
rec_idx = 0
file_idx = 0
rec_this_file = 0
pgn = open(pgnfile, encoding='utf-8-sig')

while True:
    game = chess.pgn.read_game(pgn)

    if (game == None) or (rec_this_file >= games_per_file):
        filename = prefix + '-' + str( file_idx ) + '.pgn.json'
        file_idx = file_idx + 1

        with open(filename, 'w') as jsonfile:
            json.dump(all_pgn, jsonfile, indent=4)

        all_pgn = []
        rec_this_file = 0
        print( filename )

    if game == None:
        break

    if game.headers['Result'] == '1/2-1/2':
        continue

    uci_moves = ''
    for move in game.mainline_moves():
         uci = chess.Move.uci( move )
         uci_moves = uci_moves + ' ' + uci

    rec = {}
    rec['id'] = prefix + "-" + str( rec_idx ) 
    rec['result'] = game.headers['Result']
    rec['moves'] = uci_moves.strip()

    all_pgn.append( rec )
    rec_idx = rec_idx + 1
    rec_this_file = rec_this_file + 1



