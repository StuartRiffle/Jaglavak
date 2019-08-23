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

def encode_position_one_hot( board ):
    
    # 6 layers for piece types, white 1, black -1
    one_hot = np.zeros( (6, 8, 8), np.int8 )

    for square in range( 64 ):
        piece_type = board.piece_type_at( square )
        if piece_type != None:
            piece_index = (piece_type - 1)
            x = chess.square_file( square )
            y = chess.square_rank( square )

            one_hot[piece_index, y, x] = 1 if (board.color_at( square ) == chess.WHITE) else -1

    return one_hot


