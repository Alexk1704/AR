import tensorflow       as tf
import numpy            as np

from tensorflow         import keras
from keras.layers       import Layer

from cl_replay.api.parsing  import Kwarg_Parser


class Custom_Layer(Layer):
    ''' Define the custom layer attributes and create some state variables which do not depend on any input shapes. '''

    def __init__(self, init_super=True, **kwargs):
        self.prefix = kwargs.get('prefix', "")
        self.parser = Kwarg_Parser(**kwargs)

        self.data_type = self.parser.add_argument('--data_type',
            type=int, default=32, choices=[32, 64], help='used data type (float32, int32 or float64, int64) for all calculations and variables (numpy and TensorFlow)')
        if self.data_type == 32:
            self.dtype_tf_float, self.dtype_tf_int = tf.float32, tf.int32
            self.dtype_np_float, self.dtype_np_int = np.float32, np.int32
        else:
            self.dtype_tf_float, self.dtype_tf_int = tf.float64, tf.int64
            self.dtype_np_float, self.dtype_np_int = np.float64, np.int64

        self.layer_name             = self.parser.add_argument('--layer_name',              type=str,   default=f'{self.prefix}Layer', help='name of this layer')
        self.sharpening_chain       = self.parser.add_argument('--sharpening_chain',        type=int,   default=[],     help = "chain of layer indices for sharpening")
        self.loss_masking           = self.parser.add_argument('--loss_masking',            type=str,   default='no',   choices=['no', 'yes'], help='allow loss masking?')

        self.reconstruction_weight  = self.parser.add_argument('--reconstruction_weight',   type=float, default=.1,     help='if sampling is active, use sharpening rate to improve samples with gradient')
        self.sharpening_rate        = self.parser.add_argument('--sharpening_rate',         type=float, default=.1,     help='if sampling is active, use sharpening rate to improve samples with gradient')
        self.sharpening_iterations  = self.parser.add_argument('--sharpening_iterations',   type=int,   default=100,    help='number of sharpening iterations')
        self.target_layer           = self.parser.add_argument('--target_layer',            type=int,   default=-1,     help='target GMM layer index for sharpening')

        if init_super: super(Custom_Layer, self).__init__(name=self.layer_name, dtype=self.dtype_tf_float)

    def build(self, input_shape):
        ''' 
        Defines variables & weights (trainables).
            * Initialize the layer state
            * Create weights that depend on the shape of the input
        '''
        pass
    
    def call(self, inputs, training=None, *args, **kwargs):
        ''' 
        Apply this layer' logic on inputs (i.e. forward pass).
            * inputs may be mini-batches
            * runs eagerly
        '''
        pass
    
    def forward(self, input_tensor):
        ''' Execute the forward pass on a tensor, some parts may run in graph-mode. '''
        pass

    def compute_output_shape(self, input_shape):    pass
    # Set external params.
    def set_parameters(self, **kwargs):             pass
    # Resets layer object for next training task.
    def reset_layer(self, **kwargs):                pass
    # Sampling given a control signal from a higher layer.
    def backwards(self, topdown, *args, **kwargs):  return topdown
    # Returns an optimizer instance for this layer.
    def get_layer_opt(self):                        return None
    # Returns a dict of lambda factors to apply to the gradients before applying them.
    def get_grad_factors(self):                     return {}
    # Returns layer specific keras.metrics object(s) for the model.
    def get_layer_metrics(self):                    return None
    # Returns logging parameters for wandb.
    def get_logging_params(self):                   return {}
    # Computes a boolean mask from an input tensor, creates a boolean mask based on this layers classification results.
    def compute_mask_(self, tensor):                return tensor
    # Applies the layers' set mask to an input tensor, if a boolean mask is present, multiplex sample losses on boolean condition & apply alpha-coefficients.
    def apply_mask_(self, tensor, mask, alpha_r, alpha_f):  return tensor
    # Set by-sample learning rates.
    def get_masking_flag(self):                     return self.loss_masking

    # sampling/sharpening
    def get_sharpening_chain(self):                 return self.sharpening_chain
    def get_target_layer(self):                     return self.target_layer
    def get_reconstruction_weight(self):            return self.reconstruction_weight
    def get_sharpening_iterations(self):            return self.sharpening_iterations
    def get_sharpening_rate(self):                  return self.sharpening_rate

    # custom layer funcs
    def get_raw_layer_loss(self):                   return None
    def set_raw_layer_loss(self, l):                self.raw_return_loss = l

    def get_layer_loss(self):                       return None
    def set_layer_loss(self, l):                    self.return_loss = l

    def get_fwd_result(self):                       return None

    def pre_train_step(self):                       pass
    def post_train_step(self):                      pass

    def __str__(self):
        ''' String representation of a layer (print all variables). '''
        max_ = len(max(vars(self).keys(), key=len))
        s = ''
        s += f'Layer: {self.layer_name}\n'
        for k, v in sorted(vars(self).items()):
            if k.startswith('_'): continue
            s += f' {k:<{max_}}:{v}\n'
        return s

    def _add_layer_prefix(self, kwargs):
        ''' Add the layer prefix if a parameter named "name" exists. '''
        if 'name' in kwargs: kwargs['name'] = self.prefix + kwargs['name']

    def _add_dtype(self, kwargs):
        ''' Add previously defined dtype_tf_float to all TF variables (if not otherwise stated). '''
        if 'dtype' not in kwargs: kwargs['dtype'] = self.dtype_tf_float

    def constant(self, *args, **kwargs):
        ''' Create a layer specific named tensorflow constant. '''
        self._add_dtype(kwargs)
        return tf.constant(*args, **kwargs)

    def variable(self, *args, **kwargs):
        ''' Create a layer specific named tensorflow variable. '''
        self._add_dtype(kwargs)
        return tf.Variable(*args, **kwargs)

    def is_layer_type(self, class_name):
        ''' 
        Test if the given class_name "is in" the layer class name.
            @param class_name: name to test if is in class name string
        '''
        return class_name in self.__class__.__name__
