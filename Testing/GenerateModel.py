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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import TensorBoard

parser = argparse.ArgumentParser(
    description = 'Convolutional network generator',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter )

parser.add_argument( 'model_name', help = 'a name for the model to generate' )

parser.add_argument( '--filters-init', metavar = 'N', type = int, help = 'number of filters on first level',        default = 128 )
parser.add_argument( '--filters-scale', metavar = 'N', type = int, help = 'scale number of filters between levels', default = 2 )

parser.add_argument( '--conv-levels', metavar = 'N', type = int, help = 'levels of convolutions',                   default = 3 )
parser.add_argument( '--conv-per-level', metavar = 'N', type = int, help = 'number of 3x3 convolutions per level',  default = 4 )
parser.add_argument( '--conv-level-scale', metavar = 'N', type = float, help = 'scale count between levels',        default = 0.5 )
parser.add_argument( '--conv-1x1', action="store_true", help = 'do a 1x1 convolution before each 3x3',              default = False )
parser.add_argument( '--conv-act', metavar = 'FUNC', help = 'activation for convolutional layers',                  default = 'relu' )
parser.add_argument( '--conv-dropout', metavar = 'N', type = float, help = 'dropout rate after level',              default = 0 )

parser.add_argument( '--skip-layers', metavar = 'N', type = int, help = 'levels of skip layers',                    default = 0 )
parser.add_argument( '--skip-inputs', action="store_true", help = 'also skip the raw inputs',                       default = False )

parser.add_argument( '--dense', metavar = 'N', type = int, help = 'number of dense layers',                         default = 2 )
parser.add_argument( '--dense-size', metavar = 'N', type = int, help = 'size of the dense layers',                  default = 512 )
parser.add_argument( '--dense-act', metavar = 'FUNC', help = 'activation function for dense layers',                default = 'relu' )
parser.add_argument( '--dense-batch-norm', action="store_true", help = 'normalize before activation',               default = False )
parser.add_argument( '--dense-dropout', metavar = 'N', type = float, help = 'dropout rate after activation',        default = 0 )

parser.add_argument( '--pooling', action="store_true", help = 'max pooling layer after each level',                 default = False )
parser.add_argument( '--pooling-prescale', action="store_true", help = 'double filters w/ 1x1 before pooling',      default = False )
parser.add_argument( '--pooling-not-last', action="store_true", help = 'disable pooling on the last level',         default = False )

parser.add_argument( '--input-shape', metavar = 'FUNC', help = 'activation for the output layer',                   default = '(6,8,8)' )
parser.add_argument( '--output-shape', metavar = 'FUNC', help = 'activation for the output layer',                  default = '1' )
parser.add_argument( '--output-act', metavar = 'FUNC', help = 'activation for the output layer',                    default = 'linear' )
parser.add_argument( '--save-json', action="store_true", help = 'export the network to a .json file',               default = False )
parser.add_argument( '--preview', action="store_true", help = 'only print the model then exit',                     default = False )

args = parser.parse_args()

print( args )


desc_filters = 0
network_desc = ''

def handle_desc_filters( filters ):
    global desc_filters, network_desc
    if filters != desc_filters:
        network_desc = network_desc + '[' + str( filters ) + '] '
        desc_filters = filters

def conv_layer( layer, kernel, filters ):
    global network_desc
    handle_desc_filters( filters )
    network_desc = network_desc + str( kernel ) + 'x '
    return Conv2D( filters, kernel_size = kernel, activation = args.conv_act, padding = 'same', data_format='channels_first' )( layer )

def sep_conv_layer( layer, kernel, filters ):
    global network_desc
    handle_desc_filters( filters )
    network_desc = network_desc + str( kernel ) + 'xs '
    return SeparableConv2D( filters, kernel_size = kernel, activation = args.conv_act, padding = 'same', data_format='channels_first' )( layer )

def dense_layer( layer ):
    global network_desc
    filters = args.dense_size
    handle_desc_filters( filters )
    network_desc = network_desc + 'dense '
    layer = Dense( filters )( layer )
    if args.dense_batch_norm:
        layer = BatchNormalization()( layer )
        network_desc = network_desc + 'norm '
    layer = Activation( args.dense_act )( layer )
    if args.dense_dropout > 0:
        layer = Dropout( args.dense_dropout )( layer )    
        network_desc = network_desc + 'dropout '
    return layer

def pool_layer( layer ):
    global network_desc
    network_desc = network_desc + 'maxpool '
    return MaxPooling2D( data_format = 'channels_first' )( layer )


def module_split( layer, filters ):

   
    half_depth = int( filters / 2 )

    a = layer
    a = conv_layer( a, 1, half_depth )

    b = layer
    b = conv_layer( b, 1, half_depth )
    b = conv_layer( b, 3, half_depth )
    
    c = layer
    c = conv_layer( c, 1, half_depth )
    c = sep_conv_layer( c, (1, 3), half_depth )
    c = sep_conv_layer( c, (3, 1), half_depth )
    c = sep_conv_layer( c, (1, 3), half_depth )
    c = sep_conv_layer( c, (3, 1), half_depth )

    layer = Concatenate( axis=1 )( [b, c] )
    layer = conv_layer( layer, 1, filters )
    return layer
    #bypass = Add()( [inputs, squeeze] )
    #return bypass


position_input = Input( shape = eval(args.input_shape), name = 'position' )
layer = position_input

skip_layers = []

filters = args.filters_init
#layer = conv_layer( layer, 1, filters )

conv_per_level = args.conv_per_level
for i in range( args.conv_levels ):
    for j in range( conv_per_level ):
        layer = conv_layer( layer, 3, filters )
        #if args.conv_1x1:
        #    layer = conv_layer( layer, 1, filters )

    filters = int( filters * args.filters_scale )

    enable_pooling = args.pooling
    last_layer = (i == args.conv_levels - 1)
    if last_layer:
        if args.pooling_not_last:
            enable_pooling = False

    if enable_pooling:
        if args.pooling_prescale:
            layer = conv_layer( layer, 1, filters )
        layer = pool_layer( layer )    

    conv_per_level = int( conv_per_level * args.conv_level_scale )

    skip_layers.append(Flatten()( layer ))

    if args.conv_dropout > 0:
        layer = Dropout( args.conv_dropout )( layer )    
        network_desc = network_desc + 'dropout ' 



# 1c313131pl


layer = Flatten()( layer )

if args.skip_inputs:
    layer = Concatenate()( [layer, Flatten()( position_input )] )

if args.skip_layers > 0:
    skip_layers = skip_layers[:-1]
    skip_layers = skip_layers[-args.skip_layers:]
    for skip in skip_layers:
        layer = Concatenate()( [layer, skip] )

for i in range( args.dense ):
    layer = dense_layer( layer )

layer = Dense( eval( args.output_shape ) )( layer )
if args.output_act != 'linear':
    layer = Activation( args.output_act )( layer )


value_output = layer
model = Model( position_input, value_output ) 

model.summary()
print( position_input.shape, ' -> ', network_desc, '-> ', value_output.shape )
print()

if args.preview:
    print( args )
    quit()

if args.save_json:
    with open( args.model_name + '.json', 'w') as f:
        f.write(model.to_json())   

model.save( args.model_name + '.h5' )

