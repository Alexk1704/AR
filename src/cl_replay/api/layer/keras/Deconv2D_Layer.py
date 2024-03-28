import sys
import tensorflow as tf

from tensorflow             import keras
from keras.layers           import Conv2DTranspose
from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log


class Deconv2D_Layer(Conv2DTranspose):
    ''' Wrapper for tensorflow.keras.layers.Reshape. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',      type=str,   default=f'{self.prefix}Layer',  help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer',     type=int,   default=[None],                 help=f'prefix integer(s) of this layer inputs')
        # keras layer specific attributes
        self.filters        = self.parser.add_argument('--filters',         type=int,   default=32, help='output space dimensionality.')
        self.activation     = self.parser.add_argument('--activation',      type=str,   default='none', choices= ['none', 'relu', 'sigmoid', 'softmax'], help='sets the activation fn.')
        if self.activation == 'none': self.activation = None
        self.use_bias       = self.parser.add_argument('--use_bias',        type=str,   default='yes', choices=['yes', 'no'], help='Whether layer uses a bias vector.')
        if self.use_bias == 'yes': self.use_bias = True
        else: self.use_bias = False
        self.kernel_size     = self.parser.add_argument('--kernel_size', nargs='+', type=int, default=(2, 2), help='height/width of 2d conv window, a single integer defines same value for all spatial dims.')
        if type(self.kernel_size) != type(0):
            if len(self.kernel_size) != 2: log.error(f'Please specify a valid kernel_size: {self.kernel_size} is not supported.'); sys.exit()
            else: self.kernel_size = tuple(self.kernel_size)
        self.strides        = self.parser.add_argument('--strides', nargs='*', type=int, default=None, help='Specifies how far the pooling window moves for each pooling step.')
        if self.strides:
            if type(self.strides) != type(0):
                if len(self.strides) != 2: log.error(f'Please specify valid strides: {self.strides} is not supported.'); sys.exit()
                else: self.strides = tuple(self.strides)
        self.padding        = self.parser.add_argument('--padding', type=str,   default='same', choices=['same', 'valid'], 
                                                        help='`"valid"` means no padding. `"same"` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.')
        self.data_format    = self.parser.add_argument('--data_format', type=str,   default='channels_last', choices=['channels_first', 'channels_last'], 
                                                        help='`"channels_first"` (batch,c,h,w). `"channels_last"` (batch,h,w,c)')

        super(Deconv2D_Layer, self).__init__(name=self.layer_name, filters=self.filters, activation=self.activation, use_bias=self.use_bias, 
                                            kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, data_format=self.data_format)
        self.trainable = True


    def get_raw_return_loss(self):
        return None

    
    def pre_train_step(self):
        pass


    def post_train_step(self):
        pass


    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
