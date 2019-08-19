import sys
import json
import chess
import chess.pgn
import zipfile
import random

# Some collections of games in .pgn files are usually huge, on the order of a GB when uncompressed
# This script preprocesses them into manageable chunks of 10k games

pgnfile = sys.argv[1]
prefix = sys.argv[2]
games_per_file = 10000

biglist = []
rec_idx = 0
file_idx = 0
rec_this_file = 0
include_draws = True

pgn = open(pgnfile, encoding='utf-8-sig')

discarded = 0
processed = 0
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
    if int( processed ) % 1000 == 0:
        print( processed )

print( len(biglist), "games loaded", discarded, "discarded" )
random.shuffle( biglist )
print( "...and shuffled!" )

while len( biglist ) > 0:
    chunk = biglist[:games_per_file]
    biglist = biglist[len(chunk):]

    filename = prefix + '-' + str( file_idx ) + '.pgn.json'
    file_idx = file_idx + 1

    with open(filename, 'w') as jsonfile:
        json.dump(chunk, jsonfile, indent=4)    
