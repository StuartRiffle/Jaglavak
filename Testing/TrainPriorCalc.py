import sys
import json
import chess
import chess.engine
import random
import numpy as np
import zipfile
import glob
import os

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam


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

# The position is represented by 6 planes of 8x8 (one plane for each piece type), plus
# a silly /extra/ plane, representing the side to move, with all values 1 or -1.
position_input = Input( shape = (8, 8, 7), name = 'position' )

# The move map is indexed by [src][dest]
movemap_input  = Input( shape = (64, 64), name = 'movemap' )

# The side to move is either 1 [white] or -1 [black] for a given position
side_to_move_input = Input( shape = (1,), name = 'side_to_move' )

num_filters = 64

if True:

    conv5 = position_input
    conv3 = position_input
    conv1 = position_input




    G = Conv2D( num_filters, kernel_size = 15, activation = 'relu', padding = 'same', data_format='channels_last' )( position_input )
    H = Conv2D( num_filters, kernel_size = 11, activation = 'relu', padding = 'same', data_format='channels_last' )( position_input )
    I = Conv2D( num_filters, kernel_size = 7, activation = 'relu', padding = 'same', data_format='channels_last' )( position_input )
    J = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_last' )( position_input )
    G = MaxPooling2D()( G )
    H = MaxPooling2D()( H )
    I = MaxPooling2D()( I )
    J = MaxPooling2D()( J )

    K = Conv2D( num_filters, kernel_size = 7, activation = 'relu', padding = 'same', data_format='channels_last' )( G )
    L = Conv2D( num_filters, kernel_size = 5, activation = 'relu', padding = 'same', data_format='channels_last' )( H )
    M = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_last' )( I )
    K = MaxPooling2D()( K )
    L = MaxPooling2D()( L )
    M = MaxPooling2D()( M )

    N = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_last' )( K )
    P = Conv2D( num_filters, kernel_size = 1, activation = 'relu', padding = 'same', data_format='channels_last' )( L )
    N = MaxPooling2D()( N )
    P = MaxPooling2D()( P )

    Q = Conv2D( num_filters, kernel_size = 1, activation = 'relu', padding = 'same', data_format='channels_last' )( N )
    Q = MaxPooling2D()( Q )

    dense = tensorflow.keras.layers.concatenate( [ 
        Flatten()( G ), 
        Flatten()( H ), 
        Flatten()( I ), 
        Flatten()( J ), 
        Flatten()( K ), 
        Flatten()( L ), 
        Flatten()( M ), 
        Flatten()( N ), 
        Flatten()( P ), 
        Flatten()( Q ), 
        Flatten()( position_input ), 
#        Flatten()( movemap_input ), 
        side_to_move_input] )

    dense = Dense( 512, activation = 'relu' )( dense )

    output_layer = Dense( 64 * 64, activation = 'softmax' )( dropout )

    # 1x1x4k -> 64x64x1, a move probability map (the OUTPUT!)
    output_array = Reshape( (64, 64) )( output_layer )

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam()

    model = Model( [position_input, movemap_input, side_to_move_input], output_array ) 
    model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

#print( "Loading model" )
#model = tensorflow.keras.models.load_model('c:/dev/Jaglavak/Testing/foo.h5' )

tensorboard = TensorBoard(log_dir='c:/temp/l12', histogram_freq=0, write_graph=True, write_images=False)

model.summary()
   
while True:
    ep_list = []
    ebm_list = []
    emm_list = []
    stm_list = []

    data_directory = "c:/dev/Jaglavak/Testing/Data/Games"
    data_files_avail = glob.glob(data_directory + '/*.pgn.json.zip')
    data_file_name = data_files_avail[random.randrange( len( data_files_avail ) )]

    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )

    print( data_file_name )
    for game in game_list:
        for times in range( 3 ):
            moves  = game['moves']
            result = game['result']

            move_list = moves.split( ' ' )

            # averaging 2 random values here to bias the choice a bit towards the midgame
            valid = len( move_list ) - 1
            move_to_use = int( (random.randrange( valid ) + random.randrange( valid )) / 2 )

            if result == '1-0':
                # make sure it's a white move
                move_to_use = move_to_use & ~1
            elif result == '0-1':
                # make sure it's a black move
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

            ep  = encode_position( board )
            ebm = encode_move( target_move )
            emm = encode_all_moves( board )

            ep_list.append( ep )
            ebm_list.append( ebm )
            emm_list.append( emm )
            stm_list.append( 1 if (board.turn == chess.WHITE) else -1  )

    position_data = np.asarray( ep_list,  np.float32 )
    movemap_data  = np.asarray( emm_list, np.float32 )
    stm_data      = np.asarray( stm_list, np.float32 )
    bestmove_data = np.asarray( ebm_list, np.float32 )

    model.fit( [position_data, movemap_data, stm_data], bestmove_data, 
        epochs = 20,
        batch_size = 10,
        validation_split = 0.12,
        verbose = 1, 
        callbacks=[tensorboard] )

    model.save('c:/dev/Jaglavak/Testing/foo.h5' )
