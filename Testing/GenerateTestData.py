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
import ChessEncodings

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

pgnfile = sys.argv[1]
suffix = sys.argv[2]

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list




game_list = load_game_list( pgnfile )
analysis_engine = chess.engine.SimpleEngine.popen_uci('c:/dev/Jaglavak/Testing/Data/Engines/stockfish-10.exe')

def analyze(board, depth):
    try:
        info = analysis_engine.analyse( board, chess.engine.Limit(depth=depth))
        eval = info["score"].pov(board.turn).score() / 100
        return eval
    except:
        return None

position_eval = {}
analysis_depth = 12
games_done = 0
dupes_skipped = 0
positions_per_file = 100000
files_done = 0

def flush_to_file():
    global position_eval
    global files_done

    position_eval_shuffled = []
    for fen, value in position_eval.items():
        position_eval_shuffled.append( (fen, value) )

    random.shuffle( position_eval_shuffled )

    position_eval = {}
    dupes_skipped = 0

    position_list = []
    value_list = []

    for fen, value in position_eval_shuffled:
        pos = chess.Board( fen )
        one_hot = ChessEncodings.encode_position_one_hot( pos )
        position_list.append( one_hot )
        value_list.append( value )

    position_data = np.asarray( position_list, np.int8 )
    value_data = np.asarray( value_list, np.float32 )

    filename = suffix + '-' + str(files_done).zfill( 2 )

    np.savez( filename + '~', 
        position_one_hot = position_data,
        position_eval = value_data,
        )

    with zipfile.ZipFile( filename + '~.npz' ) as zin:
        with zipfile.ZipFile( filename + '.npz', 'w', zipfile.ZIP_DEFLATED ) as zout:
            for info in zin.infolist():
                content = zin.read( info.filename )
                zout.writestr( info.filename, content )

    os.remove( filename + '~.npz' )

    print( "*** Saved", filename)
    files_done = files_done + 1



start_time = time.time()




def looks_like_endgame(board):
    non_pawns = [0, 0]
    total_pieces = 0
    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            color = 0 if board.piece_at( square ).color == chess.WHITE else 1
            total_pieces = total_pieces + 1
            if piece_type != chess.PAWN:
                non_pawns[color] = non_pawns[color] + 1

    if (non_pawns[0] <= 2) and (non_pawns[1] <= 2) and (total_pieces <= 8):
        return True

    if board.peek().promotion != None:
        return True

    return False



for game in game_list:

    moves  = game['moves']
    move_list = moves.split( ' ' )

    value_prev = 0
    repeats = 0
    verbose = games_done % 100 == 0

    board = chess.Board()
    for move in move_list:

        try:
            board.push_uci( move )
        except:
            pass

        if looks_like_endgame( board ):
            break

        fen = board.fen()
        if fen in position_eval:
            dupes_skipped = dupes_skipped + 1
            continue

        value = analyze( board, analysis_depth )
        if value == None:
            continue

        if board.turn == chess.BLACK:
            value = -value

        if value == value_prev:
            repeats = repeats + 1
            if repeats > 10:
                # This game is going nowhere
                break
        else:
            repeats = 0
        
        value_prev = value

        winning_probability = 1 / (1 + pow( 10, -value / 4 ))
        position_eval[fen] = winning_probability

        if verbose:
            print( "Eval", value, "after", move, "wp", str( int( winning_probability * 100 ) ) + "%" )

    if verbose:
        print( board )

    games_done = games_done + 1
    if True:#:
        elapsed = time.time() - start_time
        hours = elapsed / (60 * 60)
        gph = games_done / hours

        print( games_done, "games done,", len( position_eval ), "positions,", dupes_skipped, "dupes,", int(gph), "g/h" )

    if len( position_eval ) >= positions_per_file:
        flush_to_file()

if len( position_eval ) > 0:
    flush_to_file()

    

exit(-1)
