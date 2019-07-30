import sys
import json
import chess
import chess.engine
import random
import numpy as np

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Flatten

jsonfile = "c:/dev/Jaglavak/Testing/Games/ccrl-4040-1051523-0.pgn.json"

def python_square_to_jag( ps ):
    x = ps % 8
    y = ps / 8
    x = 7 - x
    return int( y * 8 + x )

def encode_position( board ):
    encoded = np.zeros( 64 * 6 + 1 )
    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            offset = (piece_type - 1) * 64

            jag_square = python_square_to_jag( square )
            offset = int( offset + jag_square )

            if board.color_at( square ) == chess.WHITE:
                encoded[offset] = 1
            elif board.color_at( square ) == chess.BLACK:
                encoded[offset] = -1

    if board.turn == chess.WHITE:
        encoded[64 * 6] = 1
    else: 
        encoded[64 * 6] = -1
    return encoded

def encode_move( move ):
    encoded = np.zeros( 64 * 64 + 4 )
    src = python_square_to_jag( move.from_square )
    dest = python_square_to_jag( move.to_square )
    encoded[src * 64 + dest] = 1

    if move.promotion == chess.KNIGHT:
        encoded[64 * 64 + 0] = 1
    if move.promotion == chess.BISHOP:
        encoded[64 * 64 + 1] = 1
    if move.promotion == chess.ROOK:
        encoded[64 * 64 + 2] = 1
    if move.promotion == chess.QUEEN:
        encoded[64 * 64 + 3] = 1

    return encoded






inputs = []
outputs = []
batch_size = 100

with open(jsonfile) as f:
    game_list = json.load( f )

while len( inputs ) < batch_size:
    idx = random.randrange( len( game_list ) )

    moves  = game_list[idx]['moves']
    result = game_list[idx]['result']

    move_list = moves.split( ' ' )

    # bias toward the midgame
    valid = len( move_list ) - 1
    move_to_use = int( (random.randrange( valid ) + random.randrange( valid )) / 2 )

    if result == '1-0':
        # make it a white move
        move_to_use = move_to_use & ~1
    elif result == '0-1':
        # make it a black move
        move_to_use = move_to_use | 1
    else:
        continue

    board = chess.Board()
    for move in range( move_to_use ):
        board.push_uci( move_list[move] )

    target_move = board.parse_uci( move_list[move_to_use] )

    if result == '1-0':
        assert( board.turn == chess.WHITE )
    elif result == '0-1':
        assert( board.turn == chess.BLACK )

    ep = encode_position( board )
    em = encode_move( target_move )

    inputs.append( ep )
    outputs.append( em )

input_data = np.asarray( inputs, np.float32 )
output_data = np.asarray( outputs, np.float32 )

X = np.array([[0,0],[0,1],[1,0],[1,1]])

model = Sequential()
model.add( Dense( 64, activation = 'relu', input_dim = input_data.shape[1] ) )
model.add( Dense( 32, activation = 'relu' ) )
model.add( Dense( output_data.shape[1], activation = 'sigmoid' ) )

model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy'] )

model.fit( input_data, output_data,
   epochs = 10,
   batch_size = 10,
   validation_split = 0.2,
   verbose = True )

# https://int8.io/chess-position-evaluation-with-convolutional-neural-networks-in-julia/ recommends this for position evaluation:
# conv, pooling, conv, inner product, inner product, binary cross entropy loss

# https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
# recommends no pooling layers


# save here

model.save('c:/dev/Jaglavak/Testing/foo.h5')

print( "Yo" )

