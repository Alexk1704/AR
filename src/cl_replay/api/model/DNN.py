import time
import numpy as np
import tensorflow as tf

from cl_replay.api.model            import Func_Model
from cl_replay.api.parsing          import Kwarg_Parser
from cl_replay.api.utils            import log, change_loglevel


class DNN(Func_Model):
    
    
    def __init__(self, inputs, outputs, name="DNN", **kwargs):
        super(DNN, self).__init__(inputs, outputs, name, **kwargs)
        self.kwargs             = kwargs

        self.adam_epsilon       = self.parser.add_argument('--adam_epsilon',    type=float, default=1e-3, help='Optimizer learning rate.')
        self.adam_beta1         = self.parser.add_argument('--adam_beta1',      type=float, default=0.9, help='ADAM beta1')
        self.adam_beta2         = self.parser.add_argument('--adam_beta2',      type=float, default=0.999, help='ADAM beta2')
        
        self.log_level          = self.parser.add_argument('--log_level',       type=str, default='DEBUG', choices=['DEBUG', 'INFO'], help='determine level for console logging.')
        change_loglevel(self.log_level)

        self.dtype_np_float = np.float32
        self.dtype_tf_float = tf.float32


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=True, steps_per_execution=None, **kwargs):
        self.all_metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='acc'), 
            tf.keras.metrics.Mean(name='step_time')
        ]
        self.opt = tf.keras.optimizers.Adam(self.adam_epsilon, self.adam_beta1, self.adam_beta1)
        self.model_params = {}

        self.supports_chkpt = False  # TODO: enable checkpointing
        self.current_task, self.test_task = 'T?', 'T?'

        super(Func_Model, self).compile(loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False), optimizer=self.opt, metrics=self.all_metrics, run_eagerly=run_eagerly)


    def train_step(self, data, **kwargs):
        xs, ys, sw = data[0], data[1], data[2]
        
        t1 = time.time()

        with tf.GradientTape(persistent=True) as tape:
            logits = self(inputs=xs, training=True)
            loss = self.compiled_loss(ys, logits, sample_weight=sw)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        del tape
        
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        
        self.all_metrics[0].update_state(ys, logits)
        self.all_metrics[-1].update_state(delta)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data, **kwargs):
        xs, ys = data[0], data[1]
        
        t1 = time.time()
        
        logits = self(inputs=xs, training=False)
        
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms

        self.all_metrics[0].update_state(ys, logits)
        self.all_metrics[-1].update_state(delta)
        
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return self.all_metrics


    def get_model_params(self):
        ''' Return a dictionary of model parameters to be tracked for an experimental evaluation via W&B. '''
        return {}