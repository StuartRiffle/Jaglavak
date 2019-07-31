import sys
import json
import chess
import chess.engine
import random
import numpy as np

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape
#from tensorflow.keras.layers.merge import concatenate

jsonfile = "c:/dev/Jaglavak/Testing/Games/ccrl-4040-1051523-0.pgn.json"

def encode_position( board ):
    encoded = np.zeros( (8, 8, 7), np.float32 )

    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)

            x = int( square % 8 )
            y = int( square / 8 )

            if board.color_at( square ) == chess.WHITE:
                encoded[x, y, piece_index] = 1
            elif board.color_at( square ) == chess.BLACK:
                encoded[x, y, piece_index] = -1

    for x in range( 8 ):
        for y in range( 8 ):
            encoded[x, y, 6] = 1 if (board.turn == chess.WHITE) else -1 

    return encoded

def encode_move( move ):
    encoded = np.zeros( (64, 64), np.float32 )
    encoded[move.from_square, move.to_square ] = 1
    return encoded

def encode_all_moves( board ):
    encoded = np.zeros( (64, 64), np.float32 )
    for move in board.legal_moves:
        encoded[move.from_square, move.to_square] = 1
    return encoded


ep_list = []
ebm_list = []
emm_list = []
stm_list = []

batch_size = 1000

with open(jsonfile) as f:
    game_list = json.load( f )

while len( ep_list ) < batch_size:
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
    ebm = encode_move( target_move )
    emm = encode_all_moves( board )

    ep_list.append( ep )
    ebm_list.append( ebm )
    emm_list.append( emm )
    stm_list.append( 1 if (board.turn == chess.WHITE) else -1  )

def convolution2d( conv, filters, kernel, pooling ):
    result = Conv2D( filters, kernel_size = (kernel, kernel), activation = 'relu' )( conv )
    result = MaxPooling( pool_size = (pooling, pooling) )( result )
    return result

position_input = Input( shape = (8, 8, 7), name = 'position' )
movemap_input  = Input( shape = (64, 64), name = 'movemap' )
side_to_move_input = Input( shape = (1,), name = 'side_to_move' )

conv = position_input
conv = Conv2D( 100, kernel_size = (5, 5), activation = 'relu' )( conv )
conv = MaxPooling2D( pool_size = (2, 2) )( conv )

conv_flat = Flatten()( conv )

dense_inputs = tensorflow.keras.layers.concatenate( [conv_flat, side_to_move_input] )
dense = Dense( 64 * 64, activation = 'relu' )( dense_inputs )
output_layer = Dense( 64 * 64, activation = 'sigmoid' )( dense )
output_array = Reshape( (64, 64) )( output_layer )

model = Model( [position_input, side_to_move_input], output_array )
model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy'] )

position_data = np.asarray( ep_list, np.float32 )
movemap_data = np.asarray( emm_list, np.float32 )
stm_data = np.asarray( stm_list, np.float32 )
bestmove_data = np.asarray( ebm_list, np.float32 )

model.fit( [position_data, stm_data], bestmove_data, 
   epochs = 1,
   batch_size = 10,
   validation_split = 0.2,
   verbose = True )

model.save('c:/dev/Jaglavak/Testing/foo.h5')
