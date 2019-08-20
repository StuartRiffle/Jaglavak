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
import time

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SeparableConv2D, Conv2D, Flatten, Multiply, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D, BatchNormalization, LeakyReLU, Lambda
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

from multiprocessing import Process


temp_directory = "c:/temp/"
model_name = 'foo.h5'#sys.argv[1]
data_directory = 'c:/cache/'#sys.argv[2]





def conv_layer( layer, kernel, filters ):
    return Conv2D( filters, kernel_size = kernel, activation = 'relu', padding = 'same', data_format='channels_first' )( layer )

def sep_conv_layer( layer, kernel, filters ):
    return SeparableConv2D( filters, kernel_size = kernel, activation = 'relu', padding = 'same', data_format='channels_first' )( layer )

def pool_layer( layer, downsample ):
    return MaxPooling2D( pool_size = downsample, data_format = 'channels_first' )( layer )

def normalize_layer( layer ):
    return BatchNormalization( axis = 1, shape = layer.shape() )

def fooceptor( input_layer, filters ):
    lane1 = conv_layer( input_layer, 1, filters )
    lane2 = conv_layer( input_layer, 1, filters )
    lane2 = conv_layer( lane2, 3, filters )
    stacked_output = Concatenate( axis = 1 )( [lane1, lane2] )
    return stacked_output

def fooceptor2( input_layer, filters ):
    levels = 1

    a = input_layer
    for _ in range( levels ):
        a = conv_layer( a, (1, 3), filters )
        a = conv_layer( a, (3, 1), filters )

    b = input_layer
    for _ in range( levels ):
        b = conv_layer( b, (3, 1),  filters )
        b = conv_layer( b, (1, 3), filters )

    layer = Concatenate( axis = 1 )( [a, b] )

    layer = conv_layer( layer, 3, filters )
    layer = conv_layer( layer, 1, filters )
    return layer


def testlayer( input_layer, filters ):
    layer = input_layer
#    layer = sep_conv( layer, filters )
    layer = conv_layer( layer, (3, 1), filters )
    layer = conv_layer( layer, (1, 3), filters )

    return layer


def generate_model( optimizer, 
    start_filters = 128, 
    scale_filters = 2,
    levels = 3, 
    level_dropout = 0.1,
    modules_per_level = 1, 
    downsample = 2,
    dense_layer_size = 1024, 
    dense_dropout = 0.1,
    skip_layers=False,
    include_position_input=False,
    ):

    position_input = Input( shape = (6, 8, 8), name = 'position' )

    layer = position_input

    all_layers = []
    if include_position_input:
        all_layers.append(Flatten()( position_input ))

    filters = start_filters
    number_of_trey = 3

    for i in range( levels ):
        for j in range( modules_per_level ):
            for k in range( number_of_trey ):
                layer = conv_layer( layer, 3, filters )
            layer = conv_layer( layer, 1, filters )
            number_of_trey = number_of_trey - 1

        layer = pool_layer( layer, downsample )
        all_layers.append( Flatten()( layer ) )
        filters = int( filters * scale_filters )

    if skip_layers:
        layer = Concatenate( axis = 1 )( all_layers )

    layer = Flatten()( layer )
    layer = Dense( 1024, activation = 'relu' )( layer )
    layer = Dense( 128, activation = 'sigmoid' )( layer )
    layer = Reshape(target_shape = (2,8,8))( layer )

    value_output = layer

    model = Model( position_input, value_output ) 
    model.compile( 
        optimizer = optimizer,
        loss = 'mean_squared_error',
        )

    with open( model_name + '.json', 'w' ) as f:
        f.write(model.to_json())    

    #model.summary()
    #quit()
    return model


def fit_model( model, inputs, outputs, 
    validation_data = None, 
    epochs = 1, 
    batch_size = 10 ):

    start_time = time.time()

    model.fit( inputs, outputs, 
        validation_data = validation_data,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.2,
        verbose = 1, 
        shuffle = True,
        #validation_freq = epochs,
        callbacks=[tensorboard] )

    elapsed = time.time() - start_time
    #print( "eval", base_eval, "elapsed", elapsed )

def normalize_move_eval_flat( eval ):
    # (n, 2, 8, 8)
    print( "normalize_move_eval_flat", eval.shape )
    for n in range( eval.shape[0] ):
        largest = float('-inf')
        smallest = float('inf')
        for z in range( 2 ):
            for y in range( 8 ):
                for x in range( 8 ):
                    samp = eval[n, z, y, x]
                    if samp != 0:
                        largest = max( largest, samp )
                        smallest = min( smallest, samp )

        if largest > smallest:
            for z in range( 2 ):
                    scale = 1 / (largest - smallest)
                    for y in range( 8 ):
                        for x in range( 8 ):
                            samp = eval[n, z, y, x]
                            if samp != 0:
                                eval[n, z, y, x] = (samp - smallest)  * scale

def position_flat_to_one_hot( flat ):
    # (n, 8, 8)
    print( "position_flat_to_one_hot", flat.shape )
    count = flat.shape[0]
    one_hot = np.zeros( (count, 6, 8, 8), np.float32 )
    for n in range( flat.shape[0] ):
        for y in range( 8 ):
            for x in range( 8 ):
                pid = round( flat[n, y, x] * 6 )
                assert( pid >= -6 and pid <= 6 )
                if pid > 0:
                    one_hot[n, pid - 1, y, x] = 1
                if pid < 0:
                    one_hot[n, -pid - 1, y, x] = -1
    return one_hot


if __name__ == '__main__':

    while True:
        logdir = 'c:/temp/tf_' + str( random.randrange( 10000 ) )
        if not os.path.exists( logdir ):
            break

    os.system("pskill tensorboard >NUL 2>NUL")
    os.system("start /MIN tensorboard --logdir=" + logdir)

    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    model = None
    try:
        model = tensorflow.keras.models.load_model( model_name )
        print( "*** Loaded model" )
    except:
        print( "*** COULD NOT LOAD MODEL!" )
        pass


    datasets_avail = []
    datasets_processed = 0
    printed_summary = False

    epochs = 1
    batch_size = 10
    learning_rate = 0.0001
    recompile = False

    optimizer=SGD( 
        lr=learning_rate, 
        decay=1e-6, 
        momentum=0.9, 
        nesterov = True )
    #optimizer=Adam(lr=learning_rate )

    while True:
        if not model:
            model = generate_model(optimizer)

        if not printed_summary:
            model.summary()
            printed_summary = True

        if recompile:
            model.compile( 
                    optimizer = sgd,
                    loss = 'mean_squared_error',
                    metrics = ['accuracy'] )

        if len( datasets_avail ) == 0:
            datasets_avail = glob.glob(data_directory + "*.npz")
            random.shuffle( datasets_avail )

        if len( datasets_avail ) == 0:
            break

        chosen = random.choice( datasets_avail )
        datasets_avail.remove( chosen )

        print( "***** Using dataset", chosen )

        try:
            archive = np.load( chosen )

            position_data = archive['position_coded']
            position_data = position_flat_to_one_hot( position_data )

            move_value_data = archive['move_value_flat']
            move_value_data = move_value_data.reshape( (move_value_data.shape[0], 2, 8, 8) )
            
            normalize_move_eval_flat( move_value_data )

            print( "Training data shape", position_data.shape, "->", move_value_data.shape )
        except:
            print( "*** NO DATA!!!" )
            quit()

        datasets_processed = datasets_processed + 1

        print( "***** ",
            "epochs/dataset", epochs, 
            "batch size", batch_size, 
            "learning rate", learning_rate,
            "sets processed", datasets_processed,
                )

        fit_model( model, position_data, move_value_data,
            epochs = epochs,
            batch_size = batch_size )

        model.save( model_name )
        print( "***** Saved model", model_name )

