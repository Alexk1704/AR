import tensorflow as tf

from tensorflow             import keras
from keras.layers           import Dense
from cl_replay.api.parsing  import Kwarg_Parser


class Dense_Layer(Dense):
    ''' Wrapper for tensorflow.keras.layers.Dense. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',      type=str,   default=f'{self.prefix}Layer',  help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer',     type=int,   default=[None],                 help=f'prefix integer(s) of this layer inputs')
        # keras layer specific attributes
        self.units          = self.parser.add_argument('--units',           type=int,   default=100, help='sets the layer units.')
        self.activation     = self.parser.add_argument('--activation',      type=str,   default='none', choices= ['none', 'relu', 'sigmoid', 'softmax', 'tanh'], help='sets the activation fn.')
        if self.activation == 'none': self.activation = None
        self.use_bias       = self.parser.add_argument('--use_bias',        type=str,   default='yes', choices=['yes', 'no'], help='Whether layer uses a bias vector.')
        if self.use_bias == 'yes': self.use_bias = True
        else: self.use_bias = False

        super(Dense_Layer, self).__init__(name=self.layer_name, units=self.units, activation=self.activation, use_bias=self.use_bias)
        self.trainable = True


    def get_raw_return_loss(self):
        return None

    
    def pre_train_step(self):
        pass


    def post_train_step(self):
        pass


    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
