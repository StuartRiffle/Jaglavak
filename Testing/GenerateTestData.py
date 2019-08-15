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
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate, MaxPooling2D, Reshape, Dropout, SpatialDropout2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adagrad

from multiprocessing import Process


temp_directory = "c:/temp/"

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

    piece_values = [0, .1, .3, .35, .5, .9, 1]
    encoded = np.zeros( (60, 8, 8), np.float32 )

    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)

            x = chess.square_file( square )
            y = chess.square_rank( square )

            # 0 - 5: white piece values
            if board.color_at( square ) == chess.WHITE:
                encoded[piece_index, x, y] = piece_values[piece_type]

            # 6 - 11: black piece values
            if board.color_at( square ) == chess.BLACK:
                encoded[6 + piece_index, x, y] = piece_values[piece_type]

            # 12 - 27: for 16 directions, distance able to move

            for move in board.legal_moves:
                if move.from_square == square:
                    direct = get_direction_index( board, square, move.to_square )
                    distance = chess.square_distance( square, move.to_square ) * 0.1
                    if distance > encoded[12 + direct, x, y]:
                        encoded[12 + direct, x, y] = distance
                    
            # 28 - 43: for 16 directions, value of prey
            attacks = board.attacks( square )
            for prey_square in attacks:
                if board.piece_at( prey_square ) != None:
                    prey_value = piece_values[board.piece_type_at( prey_square )]
                    direct = get_direction_index( board, square, prey_square )
                    encoded[28 + direct, x, y] = prey_value

            # 44-59: for 16 directions, value of attacker
            attackers = board.attackers( board.turn, square )
            for attacker_square in attackers:
                if board.piece_at( attacker_square ) != None:
                    attacker_value = piece_values[board.piece_type_at( attacker_square )]
                    direct = get_direction_index( board, attacker_square, square )
                    encoded[44 + direct, x, y] = attacker_value

    return encoded

def encode_move( move ):
    src  = np.zeros( 64, np.float32 )
    dest = np.zeros( 64, np.float32 )
    src[move.from_square] = 1;
    dest[move.to_square] = 1;
    return src, dest

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list


ep_list = []
ebm_src_list = []
ebm_dest_list = []

data_directory = "Data/Games"
data_files_avail = glob.glob(data_directory + '/ccrl-4040-*.pgn.json.zip')

for i in range( 10 ):
    chosen_idx = random.randrange( len( data_files_avail ) )
    data_file_name = data_files_avail[chosen_idx]
    data_files_avail.remove( data_file_name )

    game_list = load_game_list( data_file_name )

    print( "*** [" + str( i ) + "] Tapping " + data_file_name )
    for game in game_list:

            moves  = game['moves']
            result = game['result']

            move_list = moves.split( ' ' )

            move_to_use = int( random.randrange( len( move_list ) - 1 ) )
            flip_board = False

            if result == '1-0':
                # make sure it's a white move
                move_to_use = move_to_use & ~1
            elif result == '0-1':
                # make sure it's a black move
                move_to_use = move_to_use | 1
                if move_to_use > len( move_list ):
                    move_to_use = move_to_use - 2
                flip_board = True
            else:
                continue

            board = chess.Board()
            for move in range( move_to_use ):
                board.push_uci( move_list[move] )

            target_move = board.parse_uci( move_list[move_to_use] )

            if flip_board:
                board = board.mirror()
                target_move = chess.Move( 
                    chess.square_mirror( target_move.from_square ), 
                    chess.square_mirror( target_move.to_square ) )

            ep  = encode_position( board )
            ebm_src, ebm_dest = encode_move( target_move )

            ep_list.append( ep )
            ebm_src_list.append( ebm_src )
            ebm_dest_list.append( ebm_dest )

position_data = np.asarray( ep_list,  np.float32 )
bestmove_src_data = np.asarray( ebm_src_list, np.float32 )
bestmove_dest_data = np.asarray( ebm_dest_list, np.float32 )

index = 0

datasets_avail = glob.glob(temp_directory + "/position-*.npy")
if len( datasets_avail ) > 0:
    datasets_avail.sort()
    highest = datasets_avail[-1]
    index_str = highest[-7:-4]
    index = int( index_str )
    index = index + 1

index_str = "{:03}".format(index)

np.save( temp_directory + "position-" + index_str, position_data )
np.save(temp_directory + "bm-src-" + index_str, bestmove_src_data )
np.save( temp_directory + "bm-dest-" + index_str, bestmove_dest_data )
print( "*** DATASET " + index_str+ " GENERATED ***")

