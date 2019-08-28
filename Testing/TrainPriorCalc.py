import sys
import json
import math
import chess
import chess.engine
import random
import numpy as np
import zipfile
import glob
import os
import datetime
import subprocess
import time
import argparse

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *#Activation, BatchNormalization, Dense, Input, SeparableConv2D, Conv2D, Flatten, Multiply, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad, RMSprop

parser = argparse.ArgumentParser(
    description = 'JAGLAVAK MODEL TRAINER',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter )

parser.add_argument( 'model', help = 'the model to train' )
parser.add_argument( 'datapath' , help = 'location of training data files' )
parser.add_argument( 'validpath' , help = 'location of validation data files' )
parser.add_argument( '--data-files', metavar = 'N', type = float, help = 'training files to use',           default = 20 )
parser.add_argument( '--val-files', metavar = 'N', type = float, help = 'validation files to use',          default = 4 )
parser.add_argument( '--val-frac', metavar = 'N', type = float, help = 'portion of validation data to use', default = 1.0 )
parser.add_argument( '--optimizer', metavar = 'NAME', help = 'keras optimizer [sgd, adam, rmsprop]',        default = 'sgd' )
parser.add_argument( '--loss', metavar = 'FUNC', help = 'keras loss function',                              default = 'mse' )
parser.add_argument( '--learning-rate', metavar = 'RATE', type = float, help = 'learning rate',             default = 0.00001 )
parser.add_argument( '--batch-size', metavar = 'SIZE', type = int, help = 'batch size',                     default = 128 )
parser.add_argument( '--momentum', metavar = 'N', type = float, help = 'momentum',                          default = 0 )
parser.add_argument( '--decay', metavar = 'N', type = int, help = 'weight decay over time',                 default = 0 )
parser.add_argument( '--epochs', metavar = 'N', type = int, help = 'passes over the training data',         default = 1 )
parser.add_argument( '--fit-size', metavar = 'N', type = int, help = 'samples per call to fit',             default = 10000 )
parser.add_argument( '--tensorboard', action="store_true", help = 'run tensorboard server',                 default = True )
args = parser.parse_args()

desc_filters = 0
network_desc = ''

def handle_desc_filters( filters ):
    global desc_filters, network_desc
    if filters != desc_filters:
        network_desc = network_desc + '[' + str( filters ) + '] '
        desc_filters = filters

def conv_layer( layer, kernel, filters ):
    global network_desc
    handle_desc_filters( filters )
    network_desc = network_desc + str( kernel ) + 'x '
    return Conv2D( filters, kernel_size = kernel, activation = 'relu', padding = 'same', data_format='channels_first' )( layer )

def dense_layer( layer, filters, dense_dropout = 0, batch_normalization = True ):
    global network_desc
    handle_desc_filters( filters )
    network_desc = network_desc + 'dense '
    layer = Dense( filters )( layer )
    if batch_normalization:
        layer = BatchNormalization()( layer )
        network_desc = network_desc + 'norm '
    layer = Activation( 'relu' )( layer )
    if dense_dropout > 0:
        layer = Dropout( dense_dropout )( layer )    
        network_desc = network_desc + 'dropout '
    return layer

def pool_layer( layer, downsample ):
    global network_desc
    network_desc = network_desc + 'pool '
    return MaxPooling2D( pool_size = downsample, data_format = 'channels_first' )( layer )


def generate_model( optimizer, 
    start_filters = 128, 
    levels = 1, 
    downsample = 2,
    dense_layers = 5,
    dense_layer_size = 1024, 
    dense_dropout = 0.5,
    skip_layers = False,
    include_position_input=False,
    modules=8,
    low_modules = 1
    ):

    position_input = Input( shape = (6, 8, 8), name = 'position' )
    layer = position_input

    '''
    all_layers = []
    if include_position_input:
        all_layers.append(Flatten()( layer ))

    filters = start_filters
    for i in range( levels ):
        num_modules = low_modules if i < levels - 1 else modules
        for j in range( num_modules ):
            layer = conv_layer( layer, 1, filters )
            layer = conv_layer( layer, 3, filters )
        #filters = filters * 2
        #layer = conv_layer( layer, 1, filters )
        if (levels > 1) and (i < levels - 1):
            layer = pool_layer( layer, downsample )    
        all_layers.append(Flatten()( layer ))

    if skip_layers:
        layer = Concatenate()( [Flatten()( layer ), all_layers] )


    #layer = conv_layer( layer, 1, int( filters / 2 ))

    for i in range( 3 ):
        filters = filters * 2
        layer = conv_layer( layer, 1, filters )
        layer = pool_layer( layer, downsample )    
    '''


    layer = Flatten()( layer )
    for i in range( dense_layers ):
        layer = dense_layer( layer, dense_layer_size, dense_dropout = dense_dropout )

    layer = Dense( 1,  activation = 'sigmoid' )( layer )

    value_output = layer

    model = Model( position_input, value_output ) 
    return model


def fit_model( model, inputs, outputs, validation_data = None ):
    start_time = time.time()
    history = model.fit( inputs, outputs, 
        validation_data = validation_data,
        epochs = 1, # we do epochs
        batch_size = args.batch_size,
        validation_split = 0, # we load it
        verbose = 1, 
        shuffle = True,
        callbacks=[tensorboard] )
    elapsed = time.time() - start_time
    print( "Total fit time", elapsed )


if __name__ == '__main__':

    while True:
        logdir = 'c:/temp/tf_' + str( random.randrange( 10000 ) )
        if not os.path.exists( logdir ):
            break

    if args.tensorboard:
        os.system("pskill tensorboard >NUL 2>NUL")
        os.system("start /MIN tensorboard --logdir=" + logdir)

    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    if args.optimizer == 'sgd':
        optimizer = SGD(
            lr = args.learning_rate, 
            decay = args.decay,
            momentum = args.momentum,
            nesterov = True )
    elif args.optimizer == 'adam':
        optimizer = Adam(
            lr = args.learning_rate, 
            decay = args.decay )
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(
            lr = args.learning_rate, 
            decay = args.decay )
    else:
        print( "Invalid optimizer" )
        exit(-1)

    model = None
    try:
        model = tensorflow.keras.models.load_model( args.model )
        print( "Loaded model", args.model )
        model.summary()
    except:
        model = generate_model( optimizer )
        model.save( args.model )
        model.summary()
        print( "Generated new model:", network_desc )

    model.compile( optimizer = optimizer, loss = args.loss )

    def load_bulk_data( path, limit, tag ):
        position_data_list = []
        eval_data_list = []
        datasets_avail = glob.glob(os.path.join(path, '*.npz'))
        random.shuffle( datasets_avail )
        for dataset in datasets_avail:
            print( tag, dataset )
            archive = np.load( dataset )
            position_data_list.append( archive['position_one_hot'] )
            eval_data_list.append( archive['position_eval'] )
            if len( eval_data_list ) >= limit:
                break
        position_data = np.concatenate( position_data_list )
        eval_data = np.concatenate( eval_data_list )
        return position_data, eval_data

    position_data, eval_data = load_bulk_data( args.datapath, args.data_files, "Training data" )
    val_position_data, val_eval_data = load_bulk_data( args.validpath, args.val_files, "Validation data" )

    val_position_data = val_position_data[:int(args.val_frac * len( val_position_data ))]
    val_eval_data = val_eval_data[:int(args.val_frac * len( val_eval_data ))]


    epoch_number = 1
    while True:
        print( "Training", position_data.shape, "->", eval_data.shape )
        print( "Starting epoch", epoch_number )
        epoch_number = epoch_number + 1

        peek_position = val_position_data[:10]
        peek_targets = val_eval_data[:10]
        print( "Validation sample target" )
        print( peek_targets )
        check_eval = model.predict( peek_position )
        print( "Validation sample predicted" )
        print( check_eval )

        chunks = math.ceil( len( position_data ) / args.fit_size )
        for i in range( chunks ):

            left = i * args.fit_size
            right = min( left + args.fit_size, len( position_data ))
            print( "Fitting samples", left, "to", right, "of",  len( position_data ) )
            fit_model( model, 
                position_data[left:right], 
                eval_data[left:right],
                validation_data = [val_position_data, val_eval_data] )

            model_checkpoint = args.model + '-' + datetime.datetime.now().strftime("%w%H")
            model.save( model_checkpoint )
            model.save( args.model )
            print( "Saved model", args.model, "checkpoint", model_checkpoint )

        if args.epochs > 0:
            if epoch_number >= args.epochs:
                break

