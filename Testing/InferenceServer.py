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
import multiprocessing

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape

loaded_models = {}
model_directory = 'c:/dev/Jaglavak/Testing'

flaskApp = flask.Flask(__name__)

def find_available_models():
    models = []
    for file in glob.glob(model_directory + '/*.h5'):
        name, ext = os.path.splitext( os.path.basename( file ) )
        models.append( name )
    return models

@flaskApp.route('/model/list')
def model_list():
    models = find_available_models()
    reply = { "models" : models }
    return flask.jsonify( reply )

@flaskApp.route('/inference', methods=['POST'])
def infer( modelname ):
    if modelname in loaded_models:
        model = loaded_models[modelname]
    else:
        model = tensorflow.keras.models.load_model( model_directory + "/" + modelname + ".h5" )
        if model:
            loaded_models[modelname] = model

    if not model:
        return "Model not found", 404

    info = json.loads( request.data )

    if info["type"] == "estimate_priors":
        fen = info["position"]
        board = chess.Board()
        board.set_board_fen( fen )

    inputs = info["inputs"]
    outputs = model.predict( inputs )
    return flask.jsonify( outputs )



if __name__ == "__main__":
    flaskApp.run()


