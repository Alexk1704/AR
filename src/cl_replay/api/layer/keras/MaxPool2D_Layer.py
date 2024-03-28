import sys
import tensorflow as tf

from tensorflow             import keras
from keras.layers           import MaxPooling2D
from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log


class MaxPool2D_Layer(MaxPooling2D):
    ''' Wrapper for tensorflow.keras.layers.Reshape. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',      type=str,   default=f'{self.prefix}Layer',  help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer',     type=int,   default=[None],                 help=f'prefix integer(s) of this layer inputs')
        # keras layer specific attributes
        self.pool_size      = self.parser.add_argument('--pool_size', nargs='+', type=int, default=(2, 2), help='window size over which to take the maximum.')
        if type(self.pool_size) != type(0):
            if len(self.pool_size) != 2: log.error(f'Please specify a valid pool_size: {self.pool_size} is not supported.'); sys.exit()
            else: self.pool_size = tuple(self.pool_size)
        self.strides        = self.parser.add_argument('--strides', nargs='*', type=int, default=None, help='Specifies how far the pooling window moves for each pooling step.')
        if self.strides:
            if type(self.strides) != type(0):
                if len(self.strides) != 2: log.error(f'Please specify valid strides: {self.strides} is not supported.'); sys.exit()
                else: self.strides = tuple(self.strides)
        self.padding        = self.parser.add_argument('--padding', type=str,   default='same', choices=['same', 'valid'], 
                                                        help='`"valid"` means no padding. `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.')
        self.data_format    = self.parser.add_argument('--data_format', type=str,   default='channels_last', choices=['channels_first', 'channels_last'], 
                                                        help='`"channels_first"` (batch,c,h,w). `"channels_last"` (batch,h,w,c)')

        super(MaxPool2D_Layer, self).__init__(name=self.layer_name, pool_size=self.pool_size, strides=self.strides, padding=self.padding, data_format=self.data_format)
        self.trainable = False


    def get_raw_return_loss(self):
        return None

    
    def pre_train_step(self):
        pass


    def post_train_step(self):
        pass


    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
