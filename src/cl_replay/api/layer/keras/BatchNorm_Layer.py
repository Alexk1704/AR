import tensorflow as tf

from tensorflow             import keras
from keras.layers           import BatchNormalization
from cl_replay.api.parsing  import Kwarg_Parser


class BatchNorm_Layer(BatchNormalization):
    ''' Wrapper for tensorflow.keras.layers.Dense. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',      type=str,   default=f'{self.prefix}Layer',  help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer',     type=int,   default=[None],                 help=f'prefix integer(s) of this layer inputs')
        
        super(BatchNorm_Layer, self).__init__(name=self.layer_name)
        self.trainable = True


    def get_raw_return_loss(self):
        return None

    
    def pre_train_step(self):
        pass


    def post_train_step(self):
        pass


    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
