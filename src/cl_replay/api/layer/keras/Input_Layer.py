import tensorflow as tf

from tensorflow         import keras
from keras.layers           import Input
from cl_replay.api.parsing  import Kwarg_Parser


class Input_Layer:
    ''' Wrapper for tensorflow.keras.layers.Input. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',  type=str, default=f'{self.prefix}Layer',    help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer', type=int, default=[None],                   help=f'prefix integer(s) of this layer inputs')
        # keras layer specific attributes
        self.shape          = self.parser.add_argument('--shape',       type=int, default=[32, 32, 1],              help='shape of 3D input tensor (H,W,C)')


    def create_obj(self):
        return Input(shape=self.shape, name=self.layer_name, dtype=tf.float32)
