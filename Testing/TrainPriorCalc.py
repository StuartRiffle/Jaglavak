import sys
import json
import chess
import chess.engine
import random
import numpy as np
import zipfile
import glob
import os
import subprocess

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

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list
    


def generate_test_data():

    ep_list = []
    ebm_list = []

    data_directory = "c:/dev/Jaglavak/Testing/Data/Games"
    data_files_avail = glob.glob(data_directory + '/*.pgn.json.zip')

    for i in range( 1 ):
        data_file_name = data_files_avail[random.randrange( len( data_files_avail ) )]
        game_list = load_game_list( data_file_name )

        print( "[" + str( i ) + "] Tap " + data_file_name )
        for game in game_list:
                moves  = game['moves']
                result = game['result']

                move_list = moves.split( ' ' )

                valid = len( move_list ) - 2
                move_to_use = int( random.randrange( valid ) )

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

                ep  = encode_position( board )
                ebm = encode_move( target_move )

                ep_list.append( ep )
                ebm_list.append( ebm )

    position_data = np.asarray( ep_list,  np.int8 )
    bestmove_data = np.asarray( ebm_list, np.int8 )

    np.save( 'c:/temp/ep', position_data )
    np.save( 'c:/temp/ebm', bestmove_data )
    print( "*** TEST DATA GENERATED ***")


# The position is represented by 6 planes of 8x8 (one plane for each piece type), plus
# a silly /extra/ plane, representing the side to move, with all values 1 or -1.
position_input = Input( shape = (7, 8, 8), name = 'position' )

num_filters = 200

def load_or_create_model():
    try:
        model = tensorflow.keras.models.load_model('c:/dev/Jaglavak/Testing/foo.h5' )
    except:
        global num_filters

        # 8x8
        conv1 = Conv2D( num_filters, kernel_size = 9, activation = 'relu', padding = 'same', data_format='channels_first' )( position_input )
        conv1 = Conv2D( num_filters, kernel_size = 5, activation = 'relu', padding = 'same', data_format='channels_first' )( conv1 )
        pool1 = MaxPooling2D( data_format='channels_first' )( conv1 )
        
        num_filters = num_filters * 2

        # 4x4
        conv2 = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_first' )( pool1 )
        conv2 = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_first' )( conv2 )
        pool2 = MaxPooling2D( data_format='channels_first' )( conv2 )

        num_filters = num_filters * 2

        # 2x2
        conv3 = Conv2D( num_filters, kernel_size = 3, activation = 'relu', padding = 'same', data_format='channels_first' )( pool2 )
        conv3 = Conv2D( num_filters, kernel_size = 1, activation = 'relu', padding = 'same', data_format='channels_first' )( conv3 )
        pool3 = MaxPooling2D( data_format='channels_first' )( conv3 )

        dense = Concatenate()( [Flatten()( pool3 ), Flatten()( position_input )] )
        dense = Dense( 2048, activation = 'relu' )( dense )

        output_layer = Dense( 64 * 64, activation = 'softmax' )( dense )
        output_array = Reshape( (64, 64) )( output_layer )

        adam = Adam()

        model = Model( position_input, output_array ) 
        model.compile( 
            optimizer = adam,
            loss = 'categorical_crossentropy', 
            metrics = ['categorical_accuracy'],
            )

    return model



if __name__ == '__main__':

    while True:
        logdir = 'c:/temp/tf_' + str( random.randrange( 10000 ) )
        if not os.path.exists( logdir ):
            break

    os.system("pskill tensorboard >NUL 2>NUL")
    os.system("start /MIN tensorboard --logdir=" + logdir)

    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)
    model = load_or_create_model()
    model.summary()

    try:
        position_test_data = np.load( "c:/temp/test100_ep.npy" )
        bestmove_test_data = np.load( "c:/temp/test100_ebm.npy" )
    except:
        pass

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
            validation_data = (position_test_data, bestmove_test_data),
            epochs = 3,
            batch_size = 100,
            validation_split = 0.1,
            verbose = 2, 
            callbacks=[tensorboard] )

        asyncgen.join()

        #foo = model.predict(position_data )

        model.save('c:/dev/Jaglavak/Testing/foo.h5' )
