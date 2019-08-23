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


def generate_model( optimizer, 
    start_filters = 128, 
    scale_filters = 2,
    add_filters = 0,
    levels = 1, 
    level_dropout = 0.1,
    modules_per_level = 7, 
    downsample = 2,
    dense_layer_size = 2048, 
    dense_dropout = 0.1,
    skip_layers=False,
    include_position_input=False,
    ):

    position_input = Input( shape = (6, 8, 8), name = 'position' )
    mask_input = Input( shape = (2, 8, 8), name = 'mask' )

    layer = position_input
    layer = Concatenate( axis = 1 )( [layer, mask_input] )

    all_layers = []
    if include_position_input:
        all_layers.append(Flatten()( layer ))

    filters = start_filters

    #layer = conv_layer( layer, 3, filters )

    for i in range( levels ):
        #layer = conv_layer( layer, 3, filters )
        for j in range( modules_per_level ):
            #layer = conv_layer( layer, 1, filters )
            layer = conv_layer( layer, 3, filters )
            filters = filters + add_filters
        modules_per_level = int( modules_per_level / 2 )

        #layer = conv_layer( layer, 1, filters )

        #if True:#i < levels - 1:
            #layer = pool_layer( layer, downsample )
        #all_layers.append( Flatten()( layer ) )
        filters = int( filters * scale_filters )

    #layer = Concatenate()( [layer, Flatten()( position_input ), Flatten()( mask_input )] )

    #filters = int( filters / 2 )

    for i in range( 3 ):
        layer = conv_layer( layer, 1, filters )
        filters = int( filters * 2 )
        layer = pool_layer( layer, downsample )

    layer = Flatten()( layer )

    if skip_layers:
        layer = Concatenate()( [layer, all_layers] )
        
    layer = Dense( 1024, activation = 'relu' )( layer )
    #layer = Dense( 1024, activation = 'relu' )( layer )
    layer = Dense( 1,  activation = 'tanh' )( layer )

    layer = Reshape(target_shape = (2,8,8))( layer )
    layer = Multiply()( [layer, mask_input] )

    value_output = layer

    model = Model( [position_input, mask_input], value_output ) 
    model.compile( 
        optimizer = optimizer,
        loss = 'mse',
        metrics = ['accuracy'], 
        )

    with open( model_name + '.json', 'w' ) as f:
        f.write(model.to_json())    

    model.summary()
    quit()
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
    learning_rate = 0.1
    recompile = False

    sgd=SGD( 
        lr=learning_rate, 
        #decay=1e-6, 
        #momentum = 0.9,
        nesterov = True,
        )
    adam=Adam(lr=learning_rate )

    while True:
        if not model:
            model = generate_model(sgd)
            model.save( model_name )

        if not printed_summary:
            model.summary()
            printed_summary = True

        if recompile:
            '''
            model.compile( 
                    optimizer = sgd,
                    loss = 'mse',
                    metrics = ['accuracy'] )
            '''

        if len( datasets_avail ) == 0:
            datasets_avail = glob.glob(data_directory + "*.npz")
            random.shuffle( datasets_avail )

        if len( datasets_avail ) == 0:
            break

        chosen = random.choice( datasets_avail )
        datasets_avail.remove( chosen )

        print( "***** Using dataset", chosen )

        archive = np.load( chosen )
        position_data = archive['position']
        move_value_data = archive['move_value_flat']

        position_data = position_data

        # HACK: fabricate an output mask
        mask_data = (move_value_data != 0).astype( np.int8 )

        print( "Training data shape", position_data.shape, "->", move_value_data.shape, "mask", mask_data.shape )

        print( "***** ",
            "epochs/dataset", epochs, 
            "batch size", batch_size, 
            "learning rate", learning_rate,
            "sets processed", datasets_processed,
            )

        fit_model( model, [position_data, mask_data], move_value_data,
            epochs = epochs,
            batch_size = batch_size )

        datasets_processed = datasets_processed + 1

        model.save( model_name )
        print( "***** Saved model", model_name )

