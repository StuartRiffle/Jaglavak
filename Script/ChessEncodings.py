import sys
import chess
import chess.engine
import random
import numpy as np

def encode_position_one_hot( board ):
    # 12 layers for piece types: PNBRQK for both the current turn and opponent
    one_hot = np.zeros( (12, 8, 8), np.int8 )
    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)
            x = chess.square_file( square )
            y = chess.square_rank( square )
            if (board.color_at( square ) == board.turn):
                one_hot[piece_index, y, x] = 1
            else:
                one_hot[piece_index + 6, y, x] = 1
    return one_hot

def valid_move_mask_from_move_map( move_map ):
    mask = np.where( move_map != 0, 1, 0 )
    return mask



