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
    src  = np.zeros( 64, np.int8 )
    dest = np.zeros( 64, np.int8 )
    src[move.from_square] = 1;
    dest[move.to_square] = 1;
    return src, dest

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list
    
def generate_test_data():

    ep_list = []
    ebm_src_list = []
    ebm_dest_list = []

    data_directory = "c:/dev/Jaglavak/Testing/Data/Games"
    data_files_avail = glob.glob(data_directory + '/ccrl-4040-*.pgn.json.zip')

    for i in range( 1 ):
        data_file_name = data_files_avail[random.randrange( len( data_files_avail ) )]
        game_list = load_game_list( data_file_name )

        print( "*** [" + str( i ) + "] Tapping " + data_file_name )
        for game in game_list:
                moves  = game['moves']
                result = game['result']

                move_list = moves.split( ' ' )

                valid = len( move_list ) - 2
                move_to_use = int( (random.randrange( valid ) + random.randrange( valid )) / 2 )
                flip_board = False

                if result == '1-0':
                    # make sure it's a white move
                    move_to_use = move_to_use & ~1
                elif result == '0-1':
                    # make sure it's a black move
                    move_to_use = move_to_use | 1
                    flip_board = True
                else:
                    continue

                board = chess.Board()
                for move in range( move_to_use ):
                    board.push_uci( move_list[move] )

                target_move = board.parse_uci( move_list[move_to_use] )

                if flip_board:
                    # Everything is pres
                    board = board.mirror()
                    target_move = Move( 
                        chess.square_mirror( target_move.from_square ), 
                        chess.square_mirror( target_move.to_square ) )

                assert( target_move in board.legal_moves() )

                ep  = encode_position( board )
                ebm_src, ebm_dest = encode_move( target_move )

                ep_list.append( ep )
                ebm_src_list.append( ebm_src )
                ebm_dest_list.append( ebm_dest )

    position_data = np.asarray( ep_list,  np.int8 )
    bestmove_src_data = np.asarray( ebm_src_list, np.int8 )
    bestmove_dest_data = np.asarray( ebm_dest_list, np.int8 )

    np.save( 'c:/temp/ep', position_data )
    np.save( 'c:/temp/ebm_src', bestmove_src_data )
    np.save( 'c:/temp/ebm_dest', bestmove_dest_data )
    print( "*** NEXT DATASET GENERATED ***")


# The position is represented by 6 planes of 8x8 (one plane for each piece type), plus
# a silly /extra/ plane, representing the side to move, with all values 1 or -1.


def conv_layer( layer, kernel, filters ):
    return Conv2D( filters, kernel_size = kernel, activation = 'relu', padding = 'same', data_format='channels_first' )( layer )

def pool_layer( layer ):
    return MaxPooling2D( data_format = 'channels_first' )( layer )

def load_or_create_model():
    try:
        model = tensorflow.keras.models.load_model('c:/dev/Jaglavak/Testing/foo.h5' )
        print( "*** MODEL LOADED ***" )
    except:
        global num_filters

        position_input = Input( shape = (7, 8, 8), name = 'position' )
        filters = 100
        layer = position_input


        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = pool_layer( layer )
        pool1 = layer

        filters = filters * 2

        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = pool_layer( layer )
        pool2 = layer

        filters = filters * 2

        layer = conv_layer( layer, 3, filters )
        layer = conv_layer( layer, 3, filters )
        layer = pool_layer( layer )
        pool3 = layer

        dense = Concatenate()( [
            Flatten()( position_input ),
            Flatten()( pool1 ), 
            Flatten()( pool2 ), 
            Flatten()( pool3 ), 
            ] )
        #dense = Flatten()( layer )
        dense = Dense( 2048, activation = 'relu' )( dense )

        output_src  = Dense( 64, activation = 'softmax' )( dense )
        output_dest = Dense( 64, activation = 'softmax' )( dense )
#        outputs = Concatenate()( [output_src, output_dest], axis = 1 )


        adam = Adam()

        model = Model( position_input, [output_src, output_dest] ) 
        model.compile( 
            optimizer = 'sgd',
            loss = 'categorical_crossentropy',
            metrics = ['categorical_accuracy'],
            )
        model.save('c:/dev/Jaglavak/Testing/foo.h5' )
        print( "*** MODEL GENERATED ***" )

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

    val_data = None
    try:
        position_test_data = np.load( "c:/temp/test10_ep.npy" )
        bestmove_test_src  = np.load( "c:/temp/test10_ebm_src.npy" )
        bestmove_test_dest = np.load( "c:/temp/test10_ebm_dest.npy" )
        val_data = (position_test_data, [bestmove_test_src, bestmove_test_dest])
    except:
        pass

    while True:
        try:
            # FIXME for testing only
            position_data = np.load( "c:/temp/ep.npy" )
            bestmove_src  = np.load( "c:/temp/ebm_src.npy" )
            bestmove_dest = np.load( "c:/temp/ebm_dest.npy" )
        except:
            generate_test_data()

        asyncgen = Process(target=generate_test_data)
        asyncgen.start()

        model.fit( position_data, [bestmove_src, bestmove_dest], 
            #validation_data = (position_test_data, [bestmove_test_src, bestmove_test_dest]),
            epochs = 2,
            batch_size = 10,
            validation_split = 0.1,
            verbose = 2, 
            shuffle = True,
            callbacks=[tensorboard] )

        asyncgen.join()

        #foo = model.predict(position_data )

        model.save('c:/dev/Jaglavak/Testing/foo.h5' )
