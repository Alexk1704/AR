import tensorflow as tf

from tensorflow             import keras
from keras.layers           import Concatenate
from cl_replay.api.parsing  import Kwarg_Parser


class Concatenate_Layer(Concatenate):
    ''' Wrapper for tensorflow.keras.layers.Concatenate. '''

    def __init__(self, **kwargs):
        self.prefix         = kwargs.get('prefix', None)
        self.parser         = Kwarg_Parser(**kwargs)
        self.layer_name     = self.parser.add_argument('--layer_name',      type=str, default=f'{self.prefix}Layer',    help='name of this layer')
        self.input_layer    = self.parser.add_argument('--input_layer',     type=int, default=[None],                   help=f'prefix integer(s) of this layer inputs')
        # keras layer specific attributes
        self.axis           = self.parser.add_argument('--axis',            type=int, default=-1, help='concatenation axis.')

        super(Concatenate_Layer, self).__init__(name=self.layer_name, axis=self.axis)
        self.trainable = False


    def backwards(self, topdown, **kwargs):
        ''' Takes a topdown signal from a layer and splits it accordingly. '''
        if hasattr(self, 'prev_shape'):
            bs          = kwargs.get('sampling_bs', 100)
            prev_Cout   = self.prev_shape[-1] - 1
            lower_      = (topdown.shape[-1] - 1) - prev_Cout
            upper_      = topdown.shape[-1] - lower_
            topdown     = tf.slice(topdown, [0, 0, 0, lower_], [bs, 1, 1, upper_])

        return topdown


    def get_raw_return_loss(self):
        return None


    def pre_train_step(self):
        pass


    def post_train_step(self):
        pass


    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
