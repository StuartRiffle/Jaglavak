import sys
import math
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

parser = argparse.ArgumentParser(
    description = 'JAGLAVAK TEST DATA GENERATOR',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@' )

print()
parser.add_argument( 'gamefile', help = 'source game data (.pgn.json)' )
parser.add_argument( 'prefix', help = 'prefix for output files')
parser.add_argument( '--engine', help = 'UCI chess engine to use',                                      default = 'stockfish')
parser.add_argument( '--output-path', metavar = 'N', help = 'destination for output files',               default = '/Jag/Training' )
parser.add_argument( '--selection-rate', metavar = 'N', type = float, help = 'fraction of moves to use',             default = 0.1 )
parser.add_argument( '--analysis-depth', metavar = 'SINZE', type = int, help = 'engine analysis depth',                     default = 9 )
parser.add_argument( '--per-file', metavar = 'N', type = int, help = 'limit output file samples',                     default = 10000 )
args = parser.parse_args()

position_eval = {}
position_move_map = {}
games_done = 0
dupes_skipped = 0
files_done = 0

def load_game_list(data_file_name):
    game_list = []
    with zipfile.ZipFile( data_file_name ) as z:
        just_name, _ = os.path.splitext( os.path.basename( data_file_name ) )
        with z.open( just_name ) as f:
            game_list = json.loads( f.read().decode("utf-8") )
    return game_list


game_list = load_game_list( args.gamefile )
analysis_engine = chess.engine.SimpleEngine.popen_uci(args.engine)

def analyze(board, depth):
    try:
        #analysis_engine.set_option( { "Clear Hash":1 } )
        info = analysis_engine.analyse( board, chess.engine.Limit(depth=depth))
        eval = info["score"].pov(board.turn).score()
        if eval != None:
            eval = eval / 100.0
        return eval 
    except:
        return None


def flush_to_file():
    global position_eval
    global position_move_map
    global files_done

    shuffled_order = list(position_eval.keys())
    random.shuffle( shuffled_order )

    shuffled_order = shuffled_order[:args.per_file]

    position_list = []
    move_map_list = []
    value_list = []

    for fen in shuffled_order:
        pos = chess.Board( fen )
        one_hot = ChessEncodings.encode_position_one_hot( pos )
        position_list.append( one_hot )
        move_map_list.append( position_move_map[fen] )
        value_list.append( position_eval[fen] )

    position_data = np.asarray( position_list, np.int8 )
    move_map_data = np.asarray( move_map_list, np.float32 )
    value_data = np.asarray( value_list, np.float32 )

    assert( move_map_data.max() < 1.1 )
    assert( move_map_data.min() > -1.1 )


    filename = args.prefix + '-' + str(files_done).zfill( 2 )

    np.savez( filename + '~', 
        position_one_hot = position_data,
        position_eval = value_data,
        position_move_map = move_map_data,
        )

    with zipfile.ZipFile( filename + '~.npz' ) as zin:
        with zipfile.ZipFile( filename + '.npz', 'w', zipfile.ZIP_DEFLATED ) as zout:
            for info in zin.infolist():
                content = zin.read( info.filename )
                zout.writestr( info.filename, content )

    os.remove( filename + '~.npz' )

    print( "*** Saved", filename)
    files_done = files_done + 1

    position_eval = {}
    position_move_map = {}
    dupes_skipped = 0



start_time = time.time()


def get_piece_value(piece):
    values = [0, 1, 3, 3, 5, 9, 100]
    return values[piece]

def is_quiet_position(board):
    if board.is_check():
        return False
    for move in board.legal_moves:
        if board.is_capture(move):
            src = board.piece_type_at( move.from_square )
            dest = board.piece_type_at( move.to_square )
            if dest != None: # could be ep
                if get_piece_value( src ) < get_piece_value( dest ):
                    return False
    return True

def pawn_advantage_to_win_prob( cp ):
    val = 1 / (1 + pow( 10, -cp / 4 ))
    assert( val >= 0 )
    assert( val <= 1 )
    return val


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

    try:
        if board.peek().promotion != None:
            return True
    except:
        pass

    return False


move_index = 0

for game in game_list:

    print('/', end = '', flush=True)

    moves  = game['moves']
    move_list = moves.split( ' ' )

    value_prev = 0
    repeats = 0

    board = chess.Board()
    for move in move_list:
        move_index = move_index + 1

        try:
            board.push_uci( move )
        except:
            print( board.fen(), "error pushing", move )
            break

        if board.turn != chess.WHITE:
            continue

        if random.random() > args.selection_rate:
            print('-', end = '', flush=True)
            continue            

        if not is_quiet_position( board ):
            print('!', end = '', flush=True)
            continue

        if looks_like_endgame( board ):
            print('E', end = '', flush=True)
            break

        fen = board.fen()
        if fen in position_eval:
            dupes_skipped = dupes_skipped + 1
            print('d', end = '', flush=True)
            continue

        stand_pat_value = analyze( board, 1 )
        if stand_pat_value == None:
            print('*', end = '', flush=True)
            continue

        blowout = 6
        if (stand_pat_value > blowout) or (stand_pat_value < -blowout):
            print('#', end = '', flush=True)
            continue

        if stand_pat_value == value_prev:
            repeats = repeats + 1
            if repeats > 10:
                break
        else:
            repeats = 0
        
        value_prev = stand_pat_value

        missing_child_data = False

        # layer 0 == from, 1 == to
        move_eval_map = np.zeros( (2, 8, 8), np.float32 )
        move_count = np.zeros( (2, 8, 8), np.int32 )


        child_move_index = 0
        for child_move in board.legal_moves:
            board.push(child_move)
            child_eval = analyze( board, args.analysis_depth )
            board.pop()
            child_move_index = child_move_index + 1

            if child_eval == None:
                missing_child_data = True
                break

            child_eval = -child_eval # opponent's pov
            delta = child_eval - stand_pat_value

            x = chess.square_file( child_move.from_square )
            y = chess.square_rank( child_move.from_square )
            move_eval_map[0, y, x] = move_eval_map[0, y, x] + delta
            move_count[0, y, x] = move_count[0, y, x] + 1

            x = chess.square_file( child_move.to_square )
            y = chess.square_rank( child_move.to_square )
            move_eval_map[1, y, x] = move_eval_map[1, y, x] + delta
            move_count[1, y, x] = move_count[1, y, x] + 1

        if missing_child_data:
            print('m', end = '', flush=True)
            #print( "Missing child data move", move_index, "child move", child_move_index)
            continue

        if child_move_index < 6:
            continue

        for z in range( 2 ):
            hi = -100
            lo =  100
            for y in range( 8 ):
                for x in range( 8 ):
                    if move_count[z, y, x] > 0:
                        avg = move_eval_map[z, y, x] / move_count[z, y, x] 
                        wp = pawn_advantage_to_win_prob( avg )
                        hi = max( hi, wp )
                        lo = min( lo, wp )
                        move_eval_map[z, y, x] = wp
            for y in range( 8 ):
                for x in range( 8 ):
                    if move_count[z, y, x] > 0:
                        wp = move_eval_map[z, y, x]
                        if (hi - lo) >= 0.0001:
                            wp = (wp - lo) / (hi - lo)
                        wp = wp * 2 - 1
                        move_eval_map[z, y, x] = wp

        assert( move_eval_map.max() < 1.01 )
        assert( move_eval_map.min() > -1.01 )

        position_move_map[fen] = move_eval_map

        winning_probability = pawn_advantage_to_win_prob( stand_pat_value )
        position_eval[fen] = winning_probability
        print('=', end = '', flush=True)


    print('/ ', end = '', flush=True)
    games_done = games_done + 1
    if games_done % 10 == 0:
        elapsed = time.time() - start_time
        hours = elapsed / (60 * 60)
        gph = games_done / hours
        print()
        print( games_done, "games done,", len( position_eval ), "positions,", dupes_skipped, "dupes,", int(gph), "g/h" )

    if len( position_eval ) >= args.per_file:
        flush_to_file()

if len( position_eval ) > 0:
    flush_to_file()

exit(-1)
