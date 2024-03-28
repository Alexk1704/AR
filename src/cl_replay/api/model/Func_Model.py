import tensorflow as tf
import numpy as np

from tensorflow             import keras
from keras.models           import Model

from cl_replay.api.parsing  import Kwarg_Parser


class Func_Model(Model):
    ''' Define a custom keras.Model using the functional API. '''

    def __init__(self, inputs, outputs, name, **kwargs):
        self.parser = Kwarg_Parser(**kwargs)

        self.data_type = self.parser.add_argument('--data_type',
            type=int, default=32, choices=[32, 64], help='used data type (float32, int32 or float64, int64) for all calculations and variables (numpy and TensorFlow)')
        if self.data_type == 32:
            self.dtype_tf_float, self.dtype_tf_int = tf.float32, tf.int32
            self.dtype_np_float, self.dtype_np_int = np.float32, np.int32
        if self.data_type == 64:
            self.dtype_tf_float, self.dtype_tf_int = tf.float64, tf.int64
            self.dtype_np_float, self.dtype_np_int = np.float64, np.int64

        self.supports_chkpt = False

        super(Func_Model, self).__init__(inputs=inputs, outputs=outputs, name=name)


    def build(self, input_shape=None):
        ''' build() is usually not needed since symbolic DAG creation is done via Functional API. '''
        super(Func_Model, self).build(input_shape)


    def train_step(self, data, **kwargs):
        ''' 
        Called by fit() & train_on_batch(). 
            * Performs a single train-step (fwd & loss calculation) on a mini-batch of samples (data), 
            * train_step overwrites what happens on call() 
        '''
        pass


    def test_step(self, data, **kwargs):
        ''' Overwrites the logic behind model.evaluate(), performs a forward step. '''
        pass


    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def set_layer_trainable(self, trainable, name=None, index=None):
        ''' Sets the trainable attribute of a layer from the model by name or index. '''
        if not name and not index: return
        try:
            if name:    layer = self.get_layer(self, name)
            elif index: layer = self.get_layer(self, index=index)
            layer.trainable = trainable
        except: raise Exception(f'Something went wrong setting the trainable attribute of the specified layer.')


    def find_layer_by_prefix(self, prefix):
        ''' Finds a layer by prefix e.g. 'L2_'. '''
        for i, layer in enumerate(self.layers):
            layer_name = layer.name
            if str(layer_name).startswith(prefix):
                return layer, i
        return None, None


    def get_layer_index_by_name(self, name):
        ''' Returns the layer index inside the self.layers structure. '''
        for i, layer in enumerate(self.layers):
            if str(layer.name).lower() == str(name).lower(): return i
