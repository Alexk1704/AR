import tensorflow as tf

from tensorflow             import keras
from keras.layers           import Reshape
from cl_replay.api.parsing  import Kwarg_Parser


class Reshape_Layer(Reshape):
    ''' Wrapper for tensorflow.keras.layers.Reshape. '''

    def __init__(self, **kwargs):
        self.prefix             = kwargs.get('prefix', None)
        self.parser             = Kwarg_Parser(**kwargs)
        self.layer_name         = self.parser.add_argument('--layer_name',      type=str,   default=f'{self.prefix}Layer',  help='name of this layer')
        self.input_layer        = self.parser.add_argument('--input_layer',     type=int,   default=[None],                 help=f'prefix integer(s) of this layer inputs')
        self.sampling_batch_size = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,               help='sampling batch size')
        self.prev_shape         = self.parser.add_argument('--prev_shape',      type=int,   default=[None],                 help='previous shape for backwards transform')
        self.target_layer       = self.parser.add_argument('--target_layer',    type=int,   default=-1,                     help='target GMM layer index for sharpening')
        self.sharpening_chain   = self.parser.add_argument('--sharpening_chain',type=int,   default=[],                     help='chain of layer indices for sharpening')
        # keras layer specific attributes
        self.target_shape       = self.parser.add_argument('--target_shape',    type=int,   default=[None],                 help='shape of 3D input tensor (H,W,C)')

        super(Reshape_Layer, self).__init__(name=self.layer_name, target_shape=self.target_shape)
        self.trainable = False

    def backwards(self, topdown, **kwargs):
        if hasattr(self, 'prev_shape'):
            convert_shape       = [self.sampling_batch_size] + list(self.prev_shape)
            topdown             = tf.reshape(topdown, shape=convert_shape)

        return topdown

    # sampling/sharpening
    def get_sharpening_chain(self):                 return self.sharpening_chain
    def get_target_layer(self):                     return self.target_layer
    def get_reconstruction_weight(self):            return self.reconstruction_weight
    def get_sharpening_iterations(self):            return self.sharpening_iterations
    def get_sharpening_rate(self):                  return self.sharpening_rate

    def reset_layer(self, **kwargs):                pass

    def get_raw_return_loss(self):                  return None
    
    def pre_train_step(self):                       pass
    def post_train_step(self):                      pass

    def is_layer_type(self, class_name):
        return class_name in self.__class__.__name__
