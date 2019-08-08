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
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

from multiprocessing import Process

def encode_position( board ):
    encoded = np.zeros( (7, 8, 8), np.int8 )

    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)

            x = 7 - int( square / 8 )
            y = int( square % 8 )

            if board.color_at( square ) == chess.WHITE:
                encoded[piece_index, x, y] = 1
            elif board.color_at( square ) == chess.BLACK:
                encoded[piece_index, x, y] = -1

    side_to_move_layer = 6

    for x in range( 8 ):
        for y in range( 8 ):
            encoded[side_to_move_layer, x, y] = 1 if (board.turn == chess.WHITE) else -1 

    return encoded

def encode_move( move ):
    encoded = np.zeros( (64, 64), np.int8 )
    encoded[move.from_square, move.to_square ] = 1
    return encoded

def encode_all_moves( board ):
    encoded = np.zeros( (64, 64), np.int8 )
    for move in board.legal_moves:
        encoded[move.from_square, move.to_square] = 1
    return encoded



def generate_test_data():

    print( "Pid", os.getpid(), "in generate_test_data")

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
        #for times in range( 3 ):
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

    position_data = np.asarray( ep_list,  np.int8 )
    #movemap_data  = np.asarray( emm_list, np.int8 )
    #stm_data      = np.asarray( stm_list, np.int8 )
    bestmove_data = np.asarray( ebm_list, np.int8 )

    np.save( 'c:/temp/ep', position_data )
    #np.save( 'c:/temp/emm', movemap_data  )
    #np.save( 'c:/temp/stm', stm_data      )
    np.save( 'c:/temp/ebm', bestmove_data )


num_filters = 200

# The position is represented by 6 planes of 8x8 (one plane for each piece type), plus
# a silly /extra/ plane, representing the side to move, with all values 1 or -1.
position_input = Input( shape = (7, 8, 8), name = 'position' )

# The move map is indexed by [src][dest]
movemap_input  = Input( shape = (64, 64), name = 'movemap' )

# The side to move is either 1 [white] or -1 [black] for a given position
side_to_move_input = Input( shape = (1,), name = 'side_to_move' )

print( "Pid", os.getpid(), "root scope before locm")


def load_or_create_model():
    print( "Pid", os.getpid(), "in load_or_create_model")
    try:
        model = tensorflow.keras.models.load_model('c:/dev/Jaglavak/Testing/foo.h5' )
    except:
        # 8x8

        conv1 = Conv2D( num_filters, kernel_size = 5, activation = 'relu', padding = 'same', data_format='channels_first' )( position_input )
        pool1 = MaxPooling2D( data_format='channels_first' )( conv1 )
        
        # Now 4x4

        num_filters = num_filters * 2

        conv2 = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_first' )( pool1 )
        pool2 = MaxPooling2D( data_format='channels_first' )( conv2 )

        # Now 2x2

        num_filters = num_filters * 2

        conv3 = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_first' )( pool2 )
        pool3 = MaxPooling2D( data_format='channels_first' )( conv3 )

        # Now 1x1

        dense = Concatenate()( [Flatten()( pool3 ), Flatten()( position_input )] )
        dense = Dense( 1024, activation = 'relu' )( dense )

        output_layer = Dense( 64 * 64, activation = 'softmax' )( dense )
        output_array = Reshape( (64, 64) )( output_layer )

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam()#lr=0.0001)
        adagrad = Adagrad()

        model = Model( position_input, output_array ) 
        model.compile( 
            #optimizer = sgd, 
            optimizer = adam,

            loss = 'categorical_crossentropy', 
            #loss = 'binary_crossentropy', 

            #metrics = ['accuracy'],
            metrics = ['categorical_accuracy'],
            )

    return model



if __name__ == '__main__':
    print( "Pid", os.getpid(), "in main")

    tensorboard = TensorBoard(log_dir='c:/temp/l16', histogram_freq=0, write_graph=True, write_images=False)
    model = load_or_create_model()
    model.summary()

    while True:
        try:
            # FIXME for testing only
            position_data = np.load( "c:/temp/ep.npy" )
            bestmove_data = np.load( "c:/temp/ebm.npy" )
        except:
            generate_test_data()

        asyncgen = Process(target=generate_test_data)
        asyncgen.start()

        model.fit( position_data, bestmove_data, 
            epochs = 10,
            batch_size = 100,
            validation_split = 0.2,
            verbose = 1, 
            callbacks=[tensorboard] )

        asyncgen.join()

        #foo = model.predict(position_data )

        model.save('c:/dev/Jaglavak/Testing/foo.h5' )
