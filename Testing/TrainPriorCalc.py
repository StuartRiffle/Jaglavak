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
from tensorflow.keras.layers import Dense, Input, SeparableConv2D, Conv2D, Flatten, Multiply, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D, BatchNormalization, LeakyReLU
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

from multiprocessing import Process


temp_directory = "c:/temp/"



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
    start_filters = 256, 
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

    #position_input = Input( shape = (12, 8, 8), name = 'position' )
    position_input = Input( shape = (12, 8, 8), name = 'position' )
    movemask_input = Input( shape = (2, 8, 8), name = 'movemask' )

    #inputs = Concatenate( axis = 1 )( [position_input, movemask_input] )
    inputs = position_input

    layer = inputs

    all_layers = []
    if include_position_input:
        all_layers.append(Flatten()( position_input ))

    filters = start_filters

    #layer = conv_layer( layer, 3, filters )
    #layer = conv_layer( layer, 3, filters )
    #layer = conv_layer( layer, 3, filters )


    for i in range( levels ):
        for j in range( modules_per_level ):
            #layer = testlayer( layer, filters )
            #layer = sep_conv_layer( layer, (1,3), filters )
            #layer = sep_conv_layer( layer, (3,1), filters )
            layer = conv_layer( layer, 3, filters )
            layer = conv_layer( layer, 3, filters )
            layer = conv_layer( layer, 1, filters )
            #if True:#i < levels - 1:
                #layer = conv_layer( layer, 1, filters * 2 )
        #modules_per_level = int( modules_per_level * 0.5 )

        layer = pool_layer( layer, downsample )

        filters = int( filters * scale_filters )
        
        all_layers.append( Flatten()( layer ) )

    if skip_layers:
        layer = Concatenate( axis = 1 )( all_layers )



    #layer = Flatten()( layer )

    layer = Flatten()( layer )

    layer = Dense( 1024, activation = 'relu' )( layer )
    layer = Dense( 1024, activation = 'relu' )( layer )
    #layer = Dropout( dense_dropout )( layer )

    layer = Dense( 128, activation = 'sigmoid' )( layer )

    layer = Reshape(target_shape = (2,8,8))( layer )
    layer = Multiply()( [layer, movemask_input] )

    movevalue_output = layer

    model = Model( [position_input, movemask_input], movevalue_output ) 
    model.compile( 
        optimizer = optimizer,
        loss = 'mean_squared_error',
        metrics = ['accuracy'],
        )

    with open('foo.json', 'w') as f:
        f.write(model.to_json())    


    model.summary()
    quit()
    return model


def fit_model( model, inputs, outputs, 
    validation_data = None, 
    epochs = 1, 
    batch_size = 10 ):

    model.fit( position_data, bestmove_src, 
        validation_data = validation_data,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0,
        verbose = 1, 
        shuffle = True,
        #validation_freq = epochs,
        callbacks=[tensorboard] )


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
        model = tensorflow.keras.models.load_model('foo.h5' )
        print( "*** Loaded model" )
    except:
        print( "*** COULD NOT LOAD MODEL!" )
        pass

    val_data_present = False
    try:
        val_position_data = np.load( "c:/temp/validation-position.npy" )
        val_bestmove_src  = np.load( "c:/temp/validation-bm-src.npy" )
        val_data_present = True
    except:
        pass


    dataset_index = -1
    printed_summary = False
    epochs = 1
    batch_size = 10
    learning_rate = 0.1
    dataset_size = 100000
    recompile = False


    datasets_processed = 0

    optimizer=SGD( lr=learning_rate, nesterov = True )

    while True:
        #for batch_size in [10, 100, 1000]:
            #for learning_rate in [0.1, 0.01, 0.001, 0.0001]:

                #optimizer = Adam(lr=learning_rate)

                if not model:
                    model = generate_model(optimizer)

                if not printed_summary:
                    model.summary()
                    printed_summary = True

                if recompile:
                    model.compile( 
                            optimizer = sgd,
                            loss = 'categorical_crossentropy',
                            metrics = ['categorical_accuracy'] )

                datasets_avail = glob.glob(temp_directory + "/position-*.npy")
                datasets_avail.sort()

                if dataset_index < 0:
                    dataset_index = random.randrange( len( datasets_avail ) )

                dataset_index = dataset_index + 1
                if dataset_index >= len( datasets_avail ):
                    dataset_index = 0
                chosen = datasets_avail[dataset_index][-16:]

                index_str = chosen[-7:-4]
                print( "Using dataset " + index_str )
                datasets_processed = datasets_processed + 1

                try:
                    position_data =np.load( temp_directory + chosen )
                    #random.shuffle( position_data )
                    position_data = position_data[:dataset_size]

                    bestmove_src  =  np.load( temp_directory + chosen.replace( "position", "bm-src" ) )
                    #random.shuffle( bestmove_src )
                    bestmove_src = bestmove_src[:dataset_size]
                    print( "*** LOADED DATASET" )
                except:
                    print( "*** NO DATA!!!" )
                    quit()










                print( "*** epochs/dataset", epochs, "batch size", batch_size, "learning rate", learning_rate )
                fit_model( model, position_data, bestmove_src,
                    epochs = epochs,
                    batch_size = batch_size )

                model.save('foo.h5' )

                if val_data_present:
                    val_cat_acc = model.evaluate( val_position_data, val_bestmove_src, batch_size=1000 )
                    print( "*** Validation result", val_cat_acc[1], "datasets processed", datasets_processed )

        #        asyncgen.join()

                #foo = model.predict(position_data )

