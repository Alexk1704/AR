import math
import numpy                as np
import tensorflow           as tf
import keras.optimizers

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log



class LOSS_FUNCTION:
    SCE     = 'softmax_cross_entropy'
    MSE     = 'mean_squared_error'
    RL      = 'q_learning'
    HUBER   = 'huber'



class Readout_Layer(Custom_Layer):


    def __init__(self, **kwargs):
        super(Readout_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs

        self.input_layer            = self.parser.add_argument('--input_layer',         type=int,   default=[None],             help='a list of prefixes of this layer inputs')
        #-------------------------- SAMPLING
        self.num_classes            = self.parser.add_argument('--num_classes',         type=int,   default=10,                 help='number of output classes')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,                help='sampling batch size')
        #-------------------------- LEARNING
        self.batch_size             = self.parser.add_argument('--batch_size',          type=int,   default=100,                help='bs')
        self.loss_function          = self.parser.add_argument('--loss_function',       type=str,   default=LOSS_FUNCTION.SCE,  help='the used loss function ["MSE" (Mean Squared Error), "SCE" (Softmax Cross Entropy), "RL (Q Learning Regression)]')
        self.scale_losses           = self.parser.add_argument('--scale_losses',        type=str,   default='no', choices=['no', 'yes'], help='scales returning loss values based on the class distribution of classes for current task')
        self.epsC                   = self.parser.add_argument('--regEps',              type=float, default=0.05,               help='layer learning rate')
        self.sgd_momentum           = self.parser.add_argument('--sgd_momentum',        type=float, default=0.,                 help='chose momentum for SGD optimization (0. = turned off)')

        self.lambda_W               = self.parser.add_argument('--lambda_W',            type=float, default=1.0,                help='adaption factor for Ws')
        self.lambda_b               = self.parser.add_argument('--lambda_b',            type=float, default=1.0,                help='adaption factor for bs')
        self.reset                  = self.parser.add_argument('--reset',               type=str,   default='no', choices=['no', 'yes'], help='(hard) reset of this layer before each sub-task?')
       
        self.wait_threshold         = self.parser.add_argument('--wait_threshold',      type=float, default=[None],         help='determines the somSigma values watched GMM(s) have to reach before allowing training (useful for higher DCGMM layers).')
        self.wait_target            = self.parser.add_argument('--wait_target',         type=str,   default=[None],         help='a list of prefixes for GMMs to watch, each prefix corresponds to a float value determined via wait.threshold.')
        self.active                 = True # controlled by Set_Model_Params if attached!


    def build(self, input_shape):
        self.channels_in    = np.prod(input_shape[1:])
        self.channels_out   = self.num_classes

        W_shape = (self.channels_in, self.channels_out)
        b_shape = [self.channels_out]

        # constants to change the adaption rate by SGD step (Ws, bs)
        self.lambda_W_factor = self.variable(0., shape=[], name='lambda_W_factor', trainable=False)
        self.lambda_b_factor = self.variable(0., shape=[], name='lambda_b_factor', trainable=False)

        init_W = tf.initializers.TruncatedNormal(stddev=1. / math.sqrt(self.channels_in))
        init_b = tf.zeros_initializer()

        self.W = self.add_weight(name='weight', shape=W_shape, initializer=init_W,     dtype=self.dtype_tf_float, trainable=True)
        self.b = self.add_weight(name='bias',   shape=b_shape, initializer=init_b,     dtype=self.dtype_tf_float, trainable=True)

        self.fwd, self.return_loss, self.raw_return_loss    = None, None, None
        self.resp_mask                                      = None

        self.build_layer_metrics()


    def call(self, inputs, training=None, *args, **kwargs):
        self.fwd = self.forward(input_tensor=inputs)

        return self.fwd


    #@tf.function(autograph=False)
    def forward(self, input_tensor):
        tensor_flattened        = tf.reshape(input_tensor, (-1, self.channels_in))
        self.logits             = tf.nn.bias_add(tf.matmul(tensor_flattened, self.W), self.b)

        return self.logits


    #@tf.function(autograph=False)
    def loss_fn(self, y_pred=None, y_true=None):
        ''' Calculate loss for the linear classifier. '''
        if y_pred is None:
            y_pred = self.fwd   # use logits from prev. fwd if they weren't passed

        if self.loss_function == LOSS_FUNCTION.SCE:
            self.raw_return_loss = -tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        elif self.loss_function == LOSS_FUNCTION.MSE:
            self.raw_return_loss = -tf.reduce_mean(((y_pred - y_true)**2), axis=1)
        elif self.loss_function == LOSS_FUNCTION.RL:
            self.raw_return_loss = tf.reduce_sum((y_pred - y_true)**2)
        elif self.loss_function == LOSS_FUNCTION.HUBER:
            self.raw_return_loss = tf.keras.losses.huber(y_pred=y_pred, y_true=y_true)
        self.raw_return_loss = self.scale_loss(y_true, self.raw_return_loss)
        
        return self.raw_return_loss


    def scale_loss(self, y_true, losses):
        ''' Scales the returning losses (per-sample) based on class frequency (based on mini-batch/dataset). '''
        if self.scale_losses == 'yes' and self.class_freq != -1:
            class_indices   = tf.argmax(y_true, axis=1)
            mask            = tf.gather(self.class_freq, indices=class_indices, axis=0)
            scaled_loss     = tf.multiply(losses, mask)
            return scaled_loss
        return losses


    def compute_mask_(self, tensor, **kwargs):
        if tensor is not None:
            return tf.equal(tf.argmax(tensor, axis=1), tf.argmax(self.fwd, axis=1))


    def apply_mask_(self, tensor, mask, alpha_r, alpha_f):
        if mask is not None:
            masked_tensor   = tf.where(mask, tensor * alpha_r, tensor * alpha_f)
            return masked_tensor
        return tensor


    def get_layer_loss(self):       return self.return_loss
    def get_raw_layer_loss(self):   return self.raw_return_loss

    def get_fwd_result(self):       return self.fwd
    def get_output_result(self):    return self.logits


    def pre_train_step(self):
        if self.active: # only learn when allowed to, ie waiting period is over (meaning observed GMM layers have reached their threshold value for somSigma)
            self.lambda_W_factor.assign(self.lambda_W)
            self.lambda_b_factor.assign(self.lambda_b)


    def reset_layer(self, **kwargs):
        ''' Reset variables W and b to their initial values (hard reset). '''
        if self.reset == 'yes':
            self.W.assign(tf.random.truncated_normal(shape=(self.channels_in, self.channels_out), stddev=(1./math.sqrt(self.channels_in))))
            self.b.assign(tf.zeros(shape=self.channels_out))
            log.debug(f'\tRESETTING {self.name} TO INITIAL VALUES...')


    def backwards(self, topdown=None, **kwargs):
        ''' 
        Performs a sampling operation.
            - topdown is a 2D tensor_like of shape [sampling_batch_size,num_classes] in one-hot! 
            - logits are created as: L = WX + b --> so X = WinvL - b. we approximate inv(W) by W.T  (1, X, Y, K)
        '''
        input_shape     = list(self.input_shape)
        input_shape[0]  = self.sampling_batch_size

        if topdown is None: return tf.ones(input_shape)

        sampling_op = tf.cast(tf.matmul(topdown - tf.expand_dims(self.b, 0), tf.transpose(self.W)), self.dtype_tf_float)
        sampling_op = tf.reshape(sampling_op - tf.reduce_min(sampling_op, axis=1, keepdims=True), input_shape)

        return sampling_op


    def compute_output_shape(self, input_shape):
        ''' Returns a tuple containing the output shape of this layers computation. '''
        return self.batch_size, self.channels_out


    def set_parameters(self, **kwargs):
        self.sigma_state    = kwargs.get('sigma_state', None)
        self.class_freq     = kwargs.get('class_freq', None)


    def get_layer_opt(self):
        ''' Returns the optimizer instance attached to this layer by the tfa.MultiOptimizer '''
        ''' 
        FIXME: test ExpDecay
        expdecay_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.epsC,
            decay_steps=20,
            decay_rate=.1)
        '''
        layer_opt = tf.keras.optimizers.SGD(learning_rate=self.epsC, momentum=self.sgd_momentum)

        return layer_opt


    def get_grad_factors(self):
        return {    
            self.W.name: self.lambda_W_factor,
            self.b.name: self.lambda_b_factor,
        }


    def build_layer_metrics(self):
        self.layer_metrics = None
        if self.loss_function != LOSS_FUNCTION.RL:
            self.layer_metrics = [
                tf.keras.metrics.Mean(name=f'{self.prefix}loss'),
                tf.keras.metrics.CategoricalAccuracy(name=f'{self.prefix}acc') # uses one-hot
            ]


    def get_layer_metrics(self):
        return self.layer_metrics


    def get_logging_params(self):
        return { 
            f'{self.prefix}epsC'           : self.epsC, 
            f'{self.prefix}loss_function'  : self.loss_function,
            f'{self.prefix}loss_masking'   : self.loss_masking,
            f'{self.prefix}lambda_W'       : self.lambda_W,
            f'{self.prefix}lambda_b'       : self.lambda_b,
            f'{self.prefix}wait_threshold' : self.wait_threshold,
            f'{self.prefix}wait_target'    : self.wait_target
        }
