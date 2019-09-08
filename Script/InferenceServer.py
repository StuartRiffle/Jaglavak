import sys
import json
import chess
import chess.engine
import random
import numpy as np
import zipfile
import flask
import io
import os
import glob
import argparse
import multiprocessing

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape

app = flask.Flask(__name__)

model_path = './'
model_ext = '.h5'
port_number = 5000

loaded_models = {}

def load_model( modelname ):
    if modelname in loaded_models:
        return loaded_models[modelname]
    try:
        model = tensorflow.keras.models.load_model( model_path + modelname + model_ext )
        loaded_models[modelname] = model
        return model
    except:
        return None


@app.route('/models')
def model_list():
    available = []
    for file in glob.glob( model_path + '*' + model_ext ):
        name, ext = os.path.splitext( os.path.basename( file ) )
        available.append( name )
    return flask.jsonify( models = available )


@app.route('/models/<modelname>')
def model_info( modelname ):
    try:
        model = load_model( modelname )
        info = json.loads( model.to_json() )
        return flask.jsonify( info )
    except:
        return flask.jsonify( error = "Model not found" )


@app.route('/models/<modelname>/infer', methods=['POST'])
def infer( modelname ):
    try:
        model = load_model( modelname )
        info = json.loads( request.data )
        inputs = info['inputs']

        if ('fen' in inputs) and not ('position' in inputs):
            board = chess.Board( inputs['fen'] )
            inputs['position'] = encode_position_one_hot( board )

        start_time = time.time()

        #################################
        outputs = model.predict( inputs )
        #################################

        response = {}
        response['time'] = time.time() - start_time
        response['outputs'] = outputs
        return flask.jsonify( response )

    except err:
        return flask.jsonify( error = str( err ) )


if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='JAGLAVAK INFERENCE SERVER' )
    parser.add_argument( '--version',
        action = 'store_true',
        dest = 'just_show_version',
        help = 'print the version and exit' )
    parser.add_argument( '--model-path', 
        dest = 'path',
        help = 'location of the serialized models (*' + model_ext + ')' )
    parser.add_argument( '--port', 
        dest = 'port',
        type = int,
        help = 'server port' )
    args = parser.parse_args()

    print()
    print( "Jaglavak Inference Server 0.0.1" )
    if args.just_show_version:
        quit()

    if args.path:
        model_path = args.path
        if len( model_path ) > 0:
            if (model_path[-1:] != '/') and (model_path[-1:] != '\\'):
                model_path = model_path + '/'

    if args.port:
        port_number = args.port

    model_path = os.path.abspath( model_path )
    print( "Serving models from", model_path, "on port", port_number )
    print()

    app.run()


