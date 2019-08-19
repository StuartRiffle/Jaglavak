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
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

from multiprocessing import Process

pgnfile = sys.argv[1]#'c:/dev/jaglavak/testing/data/games/ccrl-40-40-1.pgn.json.zip'#sys.argv[1]
suffix = sys.argv[2]#'test'#sys.argv[2]

def get_direction_index( board, from_square, to_square ):
    x1 = chess.square_file( from_square )
    y1 = chess.square_rank( from_square )
    x2 = chess.square_file( to_square )
    y2 = chess.square_rank( to_square )

    pt = board.piece_type_at( from_square )

    if pt != chess.KNIGHT:
        # 0-7 clockwise starting at N
        if y1 < y2 and x1 == x2:
            return 0
        if y1 < y2 and x1 < x2:
            return 1
        if y1 == y2 and x1 < x2:
            return 2
        if y1 > y2 and x1 < x2:
            return 3
        if y1 > y2 and x1 == x2:
            return 4
        if y1 > y2 and x1 > x2:
            return 5
        if y1 == y2 and x1 > x2:
            return 6
        if y1 < y2 and x1 > x2:
            return 7
    else:
        # knight moves clockwise
        if (x2 == x1 + 1) and (y2 == y1 + 2):
            return 8
        if (x2 == x1 + 2) and (y2 == y1 + 1):
            return 9
        if (x2 == x1 + 2) and (y2 == y1 - 1):
            return 10
        if (x2 == x1 + 1) and (y2 == y1 - 2):
            return 11
        if (x2 == x1 - 1) and (y2 == y1 - 2):
            return 12
        if (x2 == x1 - 2) and (y2 == y1 - 1):
            return 13
        if (x2 == x1 - 2) and (y2 == y1 + 1):
            return 14
        if (x2 == x1 - 1) and (y2 == y1 + 2):
            return 15

    return -1


def encode_position( board ):

    # 12 6 layers for white pieces, 6 layers for black, one hot
    sparse = np.zeros( (12, 8, 8), np.int8 )

    # 1 layer piece types 1-6, normalized 0-1, black is negative
    piece_types = np.zeros( (8, 8), np.int8 )

    

    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)

            x = chess.square_file( square )
            y = chess.square_rank( square )

            # 0 - 5: white piece values
            if board.color_at( square ) == chess.WHITE:
                sparse[piece_index, x, y] = 1
                piece_types[x, y] = piece_type / 6 

            # 6 - 11: black piece values
            if board.color_at( square ) == chess.BLACK:
                sparse[6 + piece_index, x, y] = 1
                piece_types[x, y] = -piece_type / 6

            '''
            # 0 - 11: for 16 directions, distance able to move
            max_distance = 14
            for move in board.legal_moves:
                if move.from_square == square:
                    direct = get_direction_index( board, square, move.to_square )
                    distance = chess.square_distance( square, move.to_square ) / max_distance
                    if distance > encoded[12 + direct, x, y]:
                        encoded[0 + direct, x, y] = distance
                    
            # 12 - 27: for 16 directions, value of prey
            attacks = board.attacks( square )
            for prey_square in attacks:
                if board.piece_at( prey_square ) != None:
                    prey_value = piece_values[board.piece_type_at( prey_square )]
                    direct = get_direction_index( board, square, prey_square )
                    encoded[12 + direct, x, y] = prey_value

            # 28 - 43: for 16 directions, value of attacker
            attackers = board.attackers( board.turn, square )
            for attacker_square in attackers:
                if board.piece_at( attacker_square ) != None:
                    attacker_value = piece_values[board.piece_type_at( attacker_square )]
                    direct = get_direction_index( board, attacker_square, square )
                    encoded[28 + direct, x, y] = attacker_value
            '''

    return sparse, piece_types

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list


ep_list = []
movemap_list = []
movesflat_list = []
movevalid_list = []
move_valid_flat_list = []
move_values_flat_list = []
coded_list = []
game_list = load_game_list( pgnfile )

stockyfish = chess.engine.SimpleEngine.popen_uci('c:/dev/Jaglavak/Testing/Data/Engines/stockfish-10.exe')

def analyze(board, depth):
    try:
        info = stockyfish.analyse( board, chess.engine.Limit(depth=depth))
        eval = info["score"].pov(board.turn).score() / 100
        return eval
    except:
        return None


for game in game_list:
    moves  = game['moves']
    result = game['result']

    move_list = moves.split( ' ' )
    move_to_use = int( random.randrange( len( move_list ) - 2 ) )

    board = chess.Board()
    for move in range( move_to_use ):
        board.push_uci( move_list[move] )

    move_valid = np.zeros( (64, 64), np.int8 )
    move_values = np.zeros( (64, 64), np.float32 )

    move_valid_flat = np.zeros( (2, 64), np.int8 )
    move_values_flat = np.zeros( (2, 64), np.float32 )

    print( board.fen(), "move", move_to_use )

    start_time = time.time()
    base_eval = analyze(board, 13)
    if base_eval == None:
        continue

    bad_game = False

    eval = 0
    for childmove in board.legal_moves:
        board.push(childmove)
        eval= analyze(board, 12)

        if eval == None:
            bad_game = True
            break

        delta = eval - base_eval
        move_values[childmove.from_square, childmove.to_square] = delta
        move_valid[childmove.from_square, childmove.to_square] = 1

        source_values = []

        move_valid_flat[0, childmove.from_square] = move_valid_flat[0, childmove.from_square] + 1
        move_valid_flat[1, childmove.to_square]   = move_valid_flat[1, childmove.to_square]   + 1

        move_values_flat[0, childmove.from_square] = move_values_flat[0, childmove.from_square] + delta
        move_values_flat[1, childmove.to_square]   = move_values_flat[1, childmove.to_square]   + delta

        board.pop()

    if bad_game:
        print( "Retrying bad " )
        continue

    for y in range(2):
        for x in range(64):
            count = move_valid_flat[y, x]
            if count > 0:
                move_valid_flat[y, x]  = 1
                move_values_flat[y, x] = move_values_flat[y, x] / count


    elapsed = time.time() - start_time
    print( "eval", base_eval, "elapsed", elapsed )


    sparse, coded  = encode_position( board )
    ep_list.append( sparse )
    coded_list.append( coded )
    movemap_list.append( move_values )
    movevalid_list.append( move_valid )

    move_valid_flat_list.append( move_valid_flat )
    move_values_flat_list.append( move_values_flat )

position_data = np.asarray( ep_list, np.int8 )
movemap_data = np.asarray( movemap_list, np.float16 )
movevalid_data = np.asarray( movevalid_list, np.int8 )
coded_data = np.asarray( coded_list, np.int8 )  
#move_valid_flat_data = np.asarray( move_valid_flat_list, np.int8 )
move_values_flat_data = np.asarray( move_values_flat_list, np.float16 )

np.save( "position-" + suffix, position_data )
np.save( "position-coded-" + suffix, coded_data )
#np.save( "move-value-" + suffix, movemap_data )
np.save( "move-value-flat-" + suffix, move_values_flat_data )
#np.save( "move-mask-" + suffix, movevalid_data )

quit()
