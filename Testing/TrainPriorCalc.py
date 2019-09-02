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
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
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
parser.add_argument( '--epochs', metavar = 'N', type = int, help = 'limit passes over the training data',   default = 1 )
parser.add_argument( '--fit-size', metavar = 'N', type = int, help = 'samples per call to fit',             default = 10000 )
parser.add_argument( '--checkpoints', action="store_true", help = 'export checkpoint once per hour',        default = False )
parser.add_argument( '--tensorboard', action="store_true", help = 'run tensorboard server',                 default = False )
parser.add_argument( '--gpu-memory', metavar = 'N', type = float, help = 'limit gpu memory usage',          default = 1.0 )

args = parser.parse_args()

def print_settings():
    print(
        args.optimizer + '/' + args.loss,
        "batch size", args.batch_size,
        'learning rate', args.learning_rate,
        'momentum', args.momentum,
        'decay', args.decay )

def limit_gpu_memory_usage( frac ): 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = frac
    tf.keras.backend.set_session( tf.Session( config = config ) )

def fit_model( model, inputs, outputs, validation_data = None, callbacks = [] ):
    start_time = time.time()
    history = model.fit( inputs, outputs, 
        validation_data = validation_data,
        epochs = 1, # we do epochs
        batch_size = args.batch_size,
        validation_split = 0, # we load it ourselves
        verbose = 1, 
        shuffle = True,
        callbacks = callbacks )
    elapsed = time.time() - start_time
    print( "Total fit time", elapsed )


if __name__ == '__main__':

    limit_gpu_memory_usage( args.gpu_memory )

    # FIXME
    while True:
        logdir = 'c:/temp/tf_' + str( random.randrange( 10000 ) )
        if not os.path.exists( logdir ):
            break

    callbacks = []

    if args.tensorboard:
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)
        callbacks.append( tensorboard )
        os.system("pskill tensorboard >NUL 2>NUL")
        os.system("start /MIN tensorboard --logdir=" + logdir)

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
        model_filename = args.model + '.h5'
        print( "Loading", model_filename )
        model = tf.keras.models.load_model( model_filename )
        model.summary()
    except:
        print( "ERROR: can't load model" )
        exit(-1)

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
            if limit > 0:
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

        fit_size = len( position_data )
        if args.fit_size > 0:
            fit_size = args.fit_size
            
        chunks = 1
        chunks = math.ceil( len( position_data ) / fit_size )

        for i in range( chunks ):
            left = i * fit_size
            right = min( left + fit_size, len( position_data ))
            print( "Fitting samples", left, "to", right, "of",  len( position_data ) )
            fit_model( model, 
                position_data[left:right], 
                eval_data[left:right],
                validation_data = [val_position_data, val_eval_data], 
                callbacks = callbacks )

            print_settings()

            model_filename = args.model + '.h5'
            model.save( model_filename )
            print( "Saved model", model_filename )

            if args.checkpoints:
                checkpoint_filename = args.model + '-' + datetime.datetime.now().strftime("%w%H") + '.h5'
                model.save( checkpoint_filename )
                print( "Saved checkpoint", checkpoint_filename )

        if args.epochs > 0:
            if epoch_number >= args.epochs:
                break

