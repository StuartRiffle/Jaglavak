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
import ChessEncodings
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad, RMSprop
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser(
    description = 'JAGLAVAK MODEL TRAINER',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@' )

print()
parser.add_argument( 'model', help = 'the model to train' )
parser.add_argument( '--model-path', metavar = 'N', help = 'location of model',               default = '/Jag/Models' )
parser.add_argument( '--log-path', metavar = 'N',  help = 'location for training logs',        default = '/Jag/Logs' )
parser.add_argument( '--data-path', metavar = 'N', help = 'location of training data',        default = '/Jag/Training' )
parser.add_argument( '--data-files', metavar = 'N', type = int, help = 'limit training files to use',     default = 100 )
parser.add_argument( '--val-path', metavar = 'N', help = 'location of validation data',       default = '/Jag/Validation' )
parser.add_argument( '--val-files', metavar = 'N', type = int, help = 'limit validation files to use',    default = 10 )
parser.add_argument( '--checkpoints', action="store_true", help = 'keep hourly checkpoints',                default = False )
parser.add_argument( '--checkpoint-path', metavar = 'N', help = 'location for checkpoints',   default = '/Jag/Checkpoints' )
parser.add_argument( '--gpu-memory', metavar = 'N', type = float, help = 'limit gpu memory usage',          default = 0.5 )
parser.add_argument( '--optimizer', metavar = 'NAME', help = 'keras optimizer [sgd, adam, rmsprop]',        default = 'adam' )
parser.add_argument( '--loss', metavar = 'FUNC', help = 'keras loss function',                              default = 'mse' )
parser.add_argument( '--learning-rate', metavar = 'RATE', type = float, help = 'learning rate',             default = 0.0001 )
parser.add_argument( '--batch-size', metavar = 'SIZE', type = int, help = 'batch size',                     default = 128 )
parser.add_argument( '--momentum', metavar = 'N', type = float, help = 'momentum',                          default = 0.9 )
parser.add_argument( '--decay', metavar = 'N', type = int, help = 'weight decay over time',                 default = 0 )
parser.add_argument( '--epochs', metavar = 'N', type = int, help = 'passes over the training data',         default = 1 )
parser.add_argument( '--fit-size', metavar = 'N', type = int, help = 'samples per call to fit',             default = 100000 )
parser.add_argument( '--tensorboard', action="store_true", help = 'spawn tensorboard server',               default = False )
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
    return history


if __name__ == '__main__':

    limit_gpu_memory_usage( args.gpu_memory )
    np.set_printoptions(precision=3)
    print()

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
        model_filename = os.path.join(args.model_path, args.model + '.h5')
        print( "Loading", model_filename )
        model = tf.keras.models.load_model( model_filename )
        model.summary()
    except:
        print( "ERROR: can't load model" )
        exit(-1)

    model.compile( optimizer = optimizer, loss = args.loss )

    def load_bulk_data( path, limit, tag ):
        position_data_list = []
        move_map_data_list = []
        mask_data_list = []
        eval_data_list = []
        datasets_avail = glob.glob(os.path.join(path, '*.npz'))
        random.shuffle( datasets_avail )
        bad_to_delete = []
        for dataset in datasets_avail:
            print( tag, dataset )
            try:
                archive = np.load( dataset )

                move_map = archive['position_move_map']
                assert( not np.isnan( move_map.any()))
                assert( move_map.min() > -1.1 )
                assert( move_map.max() < 1.1 )
                move_map_data_list.append( move_map )

                position_data_list.append( archive['position_one_hot'] )
                assert( not np.isnan( position_data_list[-1].any()))

                eval_data_list.append( archive['position_eval'] )
                assert( not np.isnan( eval_data_list[-1].any()))

                mask = ChessEncodings.valid_move_mask_from_move_map( move_map )
                mask_data_list.append( mask )

            except:
                print( "EXCEPTION loading", dataset )
                bad_to_delete.append( dataset )
                #raise
                pass

            if limit > 0:
                if len( eval_data_list ) >= limit:
                    break

        for f in bad_to_delete:
            os.remove( f )
                
        position_data = np.concatenate( position_data_list )
        move_map_data = np.concatenate( move_map_data_list )
        mask_data = np.concatenate( mask_data_list )
        eval_data = np.concatenate( eval_data_list )

        print( "position_data", position_data.min(), "to", position_data.max())
        print( "move_map_data", move_map_data.min(), "to", move_map_data.max())
        print( "mask_data", mask_data.min(), "to", mask_data.max())
        print( "eval_data", eval_data.min(), "to", eval_data.max())

        return position_data, move_map_data, mask_data, eval_data

    # FIXME
    position_data, move_map_data, mask_data, eval_data = load_bulk_data( args.data_path, args.data_files, "Training data" )

    val_position_data, val_move_map_data, val_mask_data, val_eval_data = load_bulk_data( args.val_path, args.val_files, "Validation data" )
    print( "Training", position_data.shape, "->", move_map_data.shape )

    best_val_loss = 1
    epoch_number = 0

    while True:
        epoch_number = epoch_number + 1
        print( "Starting epoch", epoch_number )

        peek_position = val_position_data[:10]
        peek_mask = val_mask_data[:10]
        peek_targets = val_move_map_data[:10]
        if False:
            print( "Validation sample target" )
            print( peek_targets )
            print( "Validation sample predicted" )
            print( check_move_map )

        check_move_map = model.predict( [peek_position, peek_mask] )
        for i in range( 1 ):
            print( "Sample", i, "target:" )
            print( peek_targets[0].astype(np.float32) )
            print( "Sample", i, "predicted:" )
            print( check_move_map[0].astype(np.float32) )


        fit_size = len( position_data )
        if args.fit_size > 0:
            fit_size = args.fit_size
            
        chunks = 1
        chunks = math.ceil( len( position_data ) / fit_size )

        for i in range( chunks ):
            left = i * fit_size
            right = min( left + fit_size, len( position_data ))
            print( "Fitting samples", left, "to", right, "of",  len( position_data ) )
            history = fit_model( model, 
                [position_data[left:right], mask_data[left:right]],
                move_map_data[left:right],
                validation_data = [[val_position_data, val_mask_data], val_move_map_data], 
                callbacks = callbacks )

            print_settings()

            if args.checkpoints:
                checkpoint_filename = os.path.join( args.checkpoint_path, args.model + '-' + datetime.now().strftime("%w%H") + '.h5' )
                model.save( checkpoint_filename )
                print( "Saved checkpoint", checkpoint_filename )

            loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]

            if best_val_loss == 1:
                best_val_loss = val_loss

            if val_loss <= best_val_loss:
                model_filename = args.model + '.h5'
                model.save( model_filename )
                print( "***** Validation loss not any worse! Saved model", model_filename )
                best_val_loss = val_loss

            log_file_name = os.path.join(args.log_path, args.model + ".log.csv")
            with open(log_file_name, "a") as log:
                info = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    args.model,
                    args.optimizer,
                    args.batch_size,
                    args.learning_rate,
                    epoch_number,
                    loss,
                    val_loss,
                ]

                line = ','.join(str(x) for x in info) + '\n'
                log.write( line )

        if args.epochs > 0:
            if epoch_number >= args.epochs:
                break

