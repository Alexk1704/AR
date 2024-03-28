import math
import numpy            as np
import tensorflow       as tf
import keras.optimizers

from importlib              import import_module

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log

from cl_replay.architecture.ar.layer.regularizer    import Regularizer_Method as RM



class Mode:
  DIAG   = 'diag'
  FULL   = 'full'


class Energy:
  LOGLIK = 'loglik'
  MC     = 'mc'


class GMM_Layer(Custom_Layer):
    ''' A Gaussian-Mixture-Model layer implemented in keras. '''
    def __init__(self, **kwargs):
        super(GMM_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs
        self.input_layer            = self.parser.add_argument('--input_layer',         type=int,   default=[None],         help=f'prefix integer(s) of this layer inputs')
        #-------------------------- GMM params
        self.K                      = self.parser.add_argument('--K',                   type=int,   default=5 ** 2,         help='number of gmm components')
        self.n                      = int(math.sqrt(self.K))
        self.mode                   = self.parser.add_argument('--mode',                type=str,   default=Mode.DIAG,      choices=['diag', 'full'], help='"diag" or "full" type of used covariance matrix')
        self.mu_init                = self.parser.add_argument('--mu_init',             type=float, default=0.01,           help='initialization value of prototypes')
        self.sigma_upper_bound      = self.parser.add_argument('--sigma_upper_bound',   type=float, default=20.,            help='the upper bound for clipping sigmas')
        self.somSigma_0             = self.parser.add_argument('--somSigma_0',          type=float, default=0.25 * math.sqrt(2 * self.K), help='only use auto initialization of somSigma_0')
        self.somSigma_inf           = self.parser.add_argument('--somSigma_inf',        type=float, default=0.01,           help='smallest sigma value for regularization')
        self.eps_0                  = self.parser.add_argument('--eps_0',               type=float, default=0.011,          help='start epsilon value (initial learning rate)')
        self.eps_inf                = self.parser.add_argument('--eps_inf',             type=float, default=0.01,           help='smallest epsilon value (learning rate) for regularization')
        self.energy                 = self.parser.add_argument('--energy',              type=str,   default=Energy.MC,      choices=['loglik', 'mc'], help='used energy function: "loglik" (standard log-likelihood loss using log-sum-exp-trick) or "mc" (MC approximation)')
        self.lambda_pi              = self.parser.add_argument('--lambda_pi',           type=float, default=0.5,            help='adaption factor for pis')
        self.lambda_mu              = self.parser.add_argument('--lambda_mu',           type=float, default=1.,             help='adaption factor for mus')
        self.lambda_sigma           = self.parser.add_argument('--lambda_sigma',        type=float, default=0.5,            help='adaption factor for sigmas')
        self.conv_mode              = self.parser.add_argument('--conv_mode',           type=str,   default='yes', choices=['no', 'yes'], help='if true, one gmm layer is used for input else, for each input patch one gmm layer is created')
        #-------------------------- SAMPLING params
        self.sampling_divisor       = self.parser.add_argument('--sampling_divisor',    type=float, default=1.,             help='divide stddev by this factor in sampling')
        self.use_pis                = self.parser.add_argument('--use_pis',             type=str,   default='no', choices=['no', 'yes'], help='use pis when sampling without topdown signal')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,            help='sampling batch size')
        self.sampling_S             = self.parser.add_argument('--sampling_S',          type=int,   default=1,              help='select best x prototypes for sampling')
        self.sampling_P             = self.parser.add_argument('--sampling_P',          type=float, default=1.,             help='power to raise topdown priors to. 1--> no_op')
        self.sampling_I             = self.parser.add_argument('--sampling_I',          type=int,   default=-1,             help='index of selected component')
        self.somSigma_sampling      = self.parser.add_argument('--somSigma_sampling',   type=str,   default='no', choices=['no', 'yes'], help='activate somSigma sampling (sample from radius).')
        #-------------------------- LEARNING params
        self.batch_size             = self.parser.add_argument('--batch_size',          type=int,   default=100,            help='bs')
        self.regularizer            = self.parser.add_argument('--regularizer',         type=str,   default=RM.DEFAULT_REG, help='"single_exp" (sliding average control) or "time_decay" (depending on iterations) regularizer type')
        self.reset_factor           = self.parser.add_argument('--reset_factor',        type=float, default=-1,             help='sets a layer-specific somSigma reset_factor')
        self.sgd_momentum           = self.parser.add_argument('--sgd_momentum',        type=float, default=0.,             help='chose momentum for SGD optimization (0. = turned off)')
        self.wait_threshold         = self.parser.add_argument('--wait_threshold',      type=float, default=[None],         help='determines the somSigma values watched GMM(s) have to reach before allowing training (useful for higher DCGMM layers).')
        self.wait_target            = self.parser.add_argument('--wait_target',         type=str,   default=[None],         help='a list of prefixes for GMMs to watch.')
        self.active                 = True # controlled by Set_Model_Params if attached!
        #-------------------------- GMM resp. manipulation
        self.mask_resp_mode         = self.parser.add_argument('--mask_resp_mode',      type=str,   default='none',         choices=['max', 'threshold', 'activity', 'none'], help='defines which masking/scaling mechanism is used for GMM responsibilities')
        self.manipulate_resp_when   = self.parser.add_argument('--manipulate_resp_when',type=str,   default='train',        choices=['train', 'test'], help='sets the timing when GMM probabilities are scaled (either on training or inference)')
        self.resp_threshold         = self.parser.add_argument('--resp_threshold',      type=float, default=0.51,           help='used for masking GMM responsibilities, only keeps responses above this threshold')
        self.log_bmu_activity       = self.parser.add_argument('--log_bmu_activity',    type=str,   default='no', choices=['no', 'yes'], help='print debug logs for this layer via Log_Protos callback (BMU activity, etc.)')


    def build(self, input_shape):
        self.h_out          = input_shape[1]
        self.w_out          = input_shape[2]
        self.c_in           = input_shape[3]
        self.c_out          = self.K

        if not isinstance(self.h_out, int): # compat
            self.h_out = self.h_out.value
            self.w_out = self.w_out.value
            self.c_in = self.c_in.value

        #-------------------------- SHAPES
        pis_shape       = [1, self.h_out, self.w_out, self.K]
        mus_shape       = [1, self.h_out, self.w_out, self.K, self.c_in]
        sigmas_shape    = None
        D_shape         = [1, self.h_out, self.w_out, self.K, self.c_in]
    
        if self.mode == Mode.DIAG: # full covariance matrices initialized to diagonal ones; diagonal entries given by sigma_upper_bound
            sigmas_shape = [1, self.h_out, self.w_out, self.K, self.c_in]

        if self.conv_mode == 'yes':
            sigmas_shape[1] = sigmas_shape[2]   = 1
            D_shape[1]      = D_shape[2]        = 1
            pis_shape[1]    = pis_shape[2]      = 1
            mus_shape[1]    = mus_shape[2]      = 1

        self.sigmas_shape = sigmas_shape

        #-------------------------- INIT
        # variables for time-varying factors: weights/selection probability (pi), centroids (mu), covariances (sigma)       # later set to:
        self.lambda_pi_factor       = self.variable(initial_value=0.,   shape=[], name='lambda_pi',     trainable=False)    # self.lambda_pi
        self.lambda_mu_factor       = self.variable(initial_value=0.,   shape=[], name='lambda_mu',     trainable=False)    # self.lambda_mu
        self.lambda_sigma_factor    = self.variable(initial_value=0.,   shape=[], name='lambda_sigma',  trainable=False)    # self.lambda_sigma

        init_pi         = tf.constant_initializer(1. / self.K)
        init_eps        = tf.constant_initializer(self.eps_0)
        init_somSigma   = tf.constant_initializer(self.somSigma_0)
        init_rand_mu    = tf.initializers.RandomUniform(-self.mu_init, +self.mu_init)

        self.tf_eps      = self.variable(initial_value=init_eps(shape=[]),               shape=[], name='eps',       trainable=False)
        self.tf_somSigma = self.variable(initial_value=init_somSigma(shape=[]),          shape=[], name='somSigma',  trainable=False)
        
        #-------------------------- WEIGHTS
        ''' the raw pis, before used they are passed through a softmax '''
        self.pis        = self.add_weight(name='pis',       shape=pis_shape,        initializer=init_pi,        dtype=self.dtype_tf_float, trainable=True)
        self.mus        = self.add_weight(name='mus',       shape=mus_shape,        initializer=init_rand_mu,   dtype=self.dtype_tf_float, trainable=True)
        self.D          = tf.constant(1.0)
        if self.mode == Mode.DIAG:
            init_sigma  = tf.constant_initializer(math.sqrt(self.sigma_upper_bound))
            self.sigmas = self.add_weight(name='sigmas',    shape=sigmas_shape,    initializer=init_sigma, dtype=self.dtype_tf_float, trainable=True)

        self.const_     = tf.constant(-self.c_in / 2. * tf.math.log(2. * math.pi))  # constant term in log probabilities

        #-------------------------- ANNEALING
        def prepare_annealing():
            ''' Generate structures for efficiently computing the time-varying smoothing filter for the annealing process. '''
            shift       = +1 if self.n % 2 == 1 else 0
            oneRow      = np.roll(np.arange(-self.n // 2 + shift, self.n // 2 + shift, dtype=np.float32), self.n // 2 + shift).reshape(self.n)
            npxGrid     = np.stack(self.n * [oneRow], axis=0)
            npyGrid     = np.stack(self.n * [oneRow], axis=1)
            npGrid      = np.array([ np.roll(npxGrid, x_roll, axis=1) ** 2 + np.roll(npyGrid, y_roll, axis=0) ** 2 for y_roll in range(self.n) for x_roll in range(self.n) ])
            self.xyGrid = tf.constant(npGrid.reshape(1, 1, 1, self.K, self.K))
        prepare_annealing()

        self.conv_masks     = tf.Variable(initial_value=self.xyGrid, trainable=False)
        self.last_som_sigma = float('inf')                              # to remember somSigma value from last iteration. If changed --> recompute filters

        #-------------------------- ANNEALING
        reg_class   = getattr(import_module(f'cl_replay.architecture.ar.layer.regularizer.{self.regularizer}'), self.regularizer)
        self.reg    = reg_class(**{**vars(self), **self.kwargs})        # instantiate given regularizer class object from "layer.regularizer" package


        self.fwd, self.return_loss, self.raw_return_loss    = None, None, None
        self.bmu_activations                                = None

        self.build_layer_metrics()


    def call(self, inputs, training=None, *args, **kwargs):
        self.fwd                = self.forward(input_tensor=inputs)                             # returns log_scores (N,pY,pX,K)
        self.resp               = self.get_output(self.fwd)                                     # produce output for the next layer, log-scores are converted to responsibilities (N,pY,pX,K)

        if self.mask_resp_mode != 'none':
            self.resp               = tf.stop_gradient(self.compute_mask_(self.resp))                       # computes masks based on GMM responsibilities used by this layer
            self.resp               = tf.stop_gradient(self.log_activations(self.resp, training=training))  # logs BMU activations
            self.resp               = self.mask_responsibilities(self.resp, training=training)              # masks/scales GMM responsibilities based on set flags

        return self.resp


    @tf.function(autograph=False)
    def forward(self, input_tensor):
        ''' raw input --> log-scores, i.e. log (p_k p_k) -> N,pY,pX,K '''
        diffs               = tf.expand_dims(input_tensor, 3) - self.mus                        # (N,pY,pX,1,D) - (1,pY,pX,K,D) -> (N,pY,pX,K,D)
        if self.mode == Mode.DIAG:
            log_det         = tf.reduce_sum(tf.math.log(self.sigmas), axis=4, keepdims=False)   # sum(1,pY,pX,K,D,axis=4) -> (1,pY,pX,K)
            sqDiffs         = diffs ** 2.0                                                      # -> N,pY,pX,K,D
            log_exp         = -0.5 * tf.reduce_sum(sqDiffs * (self.sigmas ** 2.), axis=4)       # sum(N,pY,pX,K,D * N,pY,pX,K,D,axis=4) -> (N,pY,pX,K)
        else: raise Exception("Non-diag cmatrix not supported. Try MFA!")
        log_probs           = (log_det + log_exp)                                               # -> N,pY,pX,K
        exp_pis             = tf.exp(self.pis)                                                  # -> 1,1,1,K
        real_pis            = exp_pis / tf.reduce_sum(exp_pis)                                  # obtain real pi values by softmax over the raw pis thus, the real pis are always positive and normalized
        log_scores          = tf.math.log(real_pis) + log_probs                                 # -> N,pY,pX,K

        return log_scores


    @tf.function(autograph=False)
    def get_output(self, log_scores):
        ''' Produce output to next layer, here: log-scores to responsibility '''
        max_logs    = tf.reduce_max(log_scores, axis=3, keepdims=True)                          # -> N,pY,pX,1
        norm_scores = tf.exp(log_scores - max_logs)                                             # -> N,pY,pX,K
        resp        = norm_scores / tf.reduce_sum(norm_scores, axis=3, keepdims=True)           # -> N,pY,pX,K

        return resp


    def compute_mask_(self, tensor):
        ''' 
        Creates a mask based on the GMM component responsibilities (self.resp) from self.call().
            - self.onehot_mask: contains only the highest activity response over all K components
            - self.gr_mask:     filter highest component response; value has to be greater than self.resp_threshold
        '''
        if tensor is not None:
            tensor_max          = tf.argmax(tensor, axis=3)
            self.onehot_mask    = tf.one_hot(tensor_max, dtype=self.dtype_tf_float, depth=self.K, axis=-1)

            #max_resp            = tf.multiply(tensor, self.onehot_mask)             # only pick entries of highest responsibility
            gr_mask             = tf.greater(tensor, self.resp_threshold)           # only returning responsbilities > threshold
            self.gr_mask        = tf.cast(gr_mask, self.dtype_tf_float)             # cast to float so we can safely multiply with resp
        return tensor


    def apply_mask_to_raw_loss(self, mask, alpha_r, alpha_f):
        ''' If a boolean mask is present, multiplex sample losses on boolean condition & apply alpha-coefficients. '''
        self.raw_return_loss    = tf.where(mask, self.raw_return_loss * alpha_r, self.raw_return_loss * alpha_f)
        self.return_loss        = -tf.reduce_mean(self.raw_return_loss)


    def mask_responsibilities(self, resp, training=None):
        ''' 
        Manipulate the GMM responsibilities.
            - max: only keep highest entry of GMM resp. matrix
            - threshold: filter GMM resp. based on a confidence threshold
            - activity: scale/norm GMM. inversely, based on their cumulative activation count
                - assumption: more than 1 GMM component is activated for one sample (non-peaking component)
                - why: GMM component connections with low activation frequencies will be overwriten faster, we want to counter this effect
                - how: can be activated either on training or testing, scale resp. of GMM components with less activity by a larger amount, while downscaling components with a high activity
        '''
        # FIXME: so far this only considers max activation f/e sample, could adapt to top-N activations, then filter
        if self.mask_resp_mode == 'max': return tf.multiply(resp, self.onehot_mask)
        elif self.mask_resp_mode == 'threshold': return tf.multiply(resp, self.gr_mask)
        elif self.mask_resp_mode == 'activity':
            if (training == True and self.manipulate_resp_when == 'train') or (
                    training == False and self.manipulate_resp_when == 'test'):
                self.comp_sum       = tf.reduce_sum(self.bmu_activations, axis=0, keepdims=True)  # 1, 1, 1, K
                count               = tf.reduce_sum(self.comp_sum)
                comp_scaling        = tf.realdiv(self.comp_sum, count)
                transformed_resp    = tf.multiply(resp, comp_scaling)
                transformed_resp    = tf.divide(transformed_resp, tf.reduce_sum(transformed_resp, axis=3, keepdims=True))
                return transformed_resp
        return resp


    def log_activations(self, resp, training=None):
        ''' Counts the GMM component activities (BMU activations) & saves them into a data structure. '''
        if (self.log_bmu_activity == 'yes') and resp.shape[0]:
            if not tf.is_tensor(self.bmu_activations):              # one time init
                self.bmu_activations = tf.Variable(lambda : tf.zeros(shape=resp.shape), dtype=self.dtype_tf_float,
                                                   name='bmu_activations', trainable=False)
            if tf.is_tensor(self.bmu_activations) and training == True:
                self.bmu_activations.assign_add(self.onehot_mask)   # accumulate BMU activations over training period
        return resp


    @tf.function(autograph=False)
    def loss_fn(self, y_pred=None, y_true=None):
        if y_pred is None:
            y_pred              = self.fwd                                                      # we use log scores from prev fwd
        log_piprobs             = tf.expand_dims(y_pred, axis=3)                                # expand3(1,pY,pX,K + N,pY,pX,K) -> N,pY,pX,1,K
        conv_log_probs          = tf.reduce_sum(log_piprobs * self.conv_masks, axis=4)          # sum4(N,pY,pX,1,K * 1,1,1,K,K) -> N,pY,pX,K
        loglikelihood_full      = tf.reduce_max(conv_log_probs, axis=3) + self.const_           # max(N,pY,pX,K) -> (N,pY,pX)
        raw_return_loss         = tf.reduce_mean(loglikelihood_full, axis=[2,1])                # N,pY,pX

        return raw_return_loss


    def get_layer_loss(self):           return self.return_loss
    def get_raw_layer_loss(self):       return self.raw_return_loss
    
    def get_fwd_result(self):           return self.fwd
    def get_output_result(self):        return self.resp


    def recompute_smoothing_filters(self, convMaskVar):
        ''' If regularizer has decreased sigma, recompute filter variable, otherwise do nothing. '''
        if self.last_som_sigma > self.tf_somSigma:
            convMaskVar.assign(tf.exp(-self.xyGrid / (2.0 * self.tf_somSigma ** 2.0)))
            convMaskVar.assign(convMaskVar / (tf.reduce_sum(convMaskVar, axis=4, keepdims=True)))
            self.last_som_sigma = self.tf_somSigma.numpy()


    def pre_train_step(self):
        self.recompute_smoothing_filters(self.conv_masks)
        if self.active: # only learn when allowed to, ie waiting period is over (meaning observed GMM layers have reached their threshold value for somSigma)
            self.set_learning_rates(self.lambda_pi, self.lambda_mu, self.lambda_sigma)


    def post_train_step(self):
        ''' Execute after each train_step, clips sigma values & adds last loss to regularizer to check limit. '''
        limit = tf.math.sqrt(self.sigma_upper_bound)
        self.sigmas.assign(tf.clip_by_value(self.sigmas, -limit, limit))

        if self.active: 
            last_loss = self.return_loss
            if tf.is_tensor(last_loss):
                last_loss = last_loss.numpy()
                self.reg.add(last_loss)
                self.reg.check_limit()


    def set_learning_rates(self, pi_factor=0., mu_factor=1., sigma_factor=0.):
        ''' Regulate learning rates for mus, pis, sigmas of a GMM_Layer. '''
        self.lambda_pi_factor.assign(pi_factor)
        self.lambda_mu_factor.assign(mu_factor)
        self.lambda_sigma_factor.assign(sigma_factor)

    
    def reset_layer(self, **kwargs): # FIXME: rename!
        ''' 
        Reset the layer by a factor to allow changes in GMM prototypes.
            - learning procedure adjusts best matching prototype(s)
            - GMMs' somSigma controls how many neighboring prototypes are affected by the learning procedure
            - small reset_factor reduces the impact on neighboring components, larger reset_factor increases this
            - semantics of reset_factor: -1 --> no_op, else: sigma --> reset_factor
        '''
        if self.reset_factor > 0.:
            self.last_som_sigma = 1000000000000.
            self.tf_somSigma.assign(self.reset_factor)
            log.debug(f'\tRESET {self.name} somSigma TO: {self.tf_somSigma.numpy()}...')
        

    def backwards(self, topdown, *args, **kwargs):
        ''' 
        Computes one backwards pass (from top to lower in model hierarchy).
            * The sampling operator creates a batch of samples from given prototypes
                - input topdown   (from upper lower): N, H, W, cOut
                - output          (to lower layer):   N, H, W, cIn
        '''
        N, h, w, cOut   = self.sampling_batch_size, self.h_out, self.w_out, self.K                                      # hierarchical procedure, samples are constructed from typical outputs of upper layer
        cIn             = self.c_in

        ''' GMM_Layer can be a top-level layer, so we need to include the case, when we have no explicit control signal '''
        if topdown is None:                                                                                             # select components, GMM prototypes can be picked over probabilities (pis)
            if self.use_pis == 'no': topdown = tf.ones([N, h, w, cOut])                                                 # e.g. pick 3: [0.8, 0.3, 0.15], 1st component contained in 80% of all samples, 2nd in 30%, etc. (sorted highest to lowest).
            else:                                                                                                       # if False  -> pis equally distributed [1., 1., 1.], each component has the same chance of being sampled
                e       = tf.exp(self.pis)                                                                              # if True   -> use model pis (learned pis)
                sm      = e / (tf.reduce_sum(e))
                topdown = tf.stack([sm for _ in range(0, self.sampling_batch_size)])

        self.sampling_placeholder   = topdown

        selectionTensor             = None
        ''' Select component indices from probabilities (I=-1: topdown values or use model probability 'pis' directly) '''
        if self.sampling_I == -1:                                                                                       # I=-1: activated top-S-sampling (default), we start from topdown & select "strongest components"
            powered = tf.pow(tf.clip_by_value(self.sampling_placeholder, 0.000001, 11000.), self.sampling_P)
            if self.sampling_S > 0:                                                                                     # S>0: sometimes a restriction can be useful, we only pick X strongest topdown values (by probabilities)
                sortedTensor    = tf.sort(powered, axis=3)
                selectionTensor = powered * tf.cast(tf.greater_equal(powered, tf.expand_dims(sortedTensor[..., -self.sampling_S], 3)), self.dtype_tf_float) # erase sub-leading ones, tf.multinomial will automatically re-normalize
                selectionTensor = tf.reshape(selectionTensor, (-1, cOut))
            else:                                                                                                       # S<=0: select components from all topdown values
                selectionTensor = tf.reshape(powered, (-1, cOut))
        #print("selectionTensor: ", selectionTensor)                                                                     # (N,K), pre-selected components showing highest activation

        if self.sampling_I == -2:                                                                                       # I=-2: cycle through selected components (equally distributed), if sampling 100 elements we sample component after component
            selectorsTensor = np.arange(0, N * h * w) % cOut
        elif self.sampling_I == -1:                                                                                     # I=-1: top-S sampling -> N * _h * _w
            selectorsTensor = tf.reshape(tf.compat.v1.random.categorical(logits=tf.math.log(selectionTensor), num_samples=1), (-1,))        # selected components to sample from (N,)
        else:                                                                                                           # I>=0: directly select components to sample from -> N * _h * _w
            selectorsTensor = tf.ones(shape=(N * h * w), dtype=self.dtype_tf_int) * self.sampling_I
        #print("pre somSigma selectorsTensor: ", selectorsTensor)                                                        # (N,), components to sample from (drawn from random categorical based on selectionTensor)

        if self.somSigma_sampling == 'yes':
            row_size = col_size = int(math.sqrt(self.K))

            row_id = selectorsTensor // row_size
            col_id = selectorsTensor % row_size

            row = tf.random.normal(
                (N*h*w,),
                mean=0.0,
                stddev=self.tf_somSigma,
                dtype=self.dtype_tf_float,
            )
            col = tf.random.normal(
                (N*h*w,),
                mean=0.0,
                stddev=self.tf_somSigma,
                dtype=self.dtype_tf_float,
            )

            row_id = (tf.round(tf.cast(row_id, dtype=self.dtype_tf_float) + row) + row_size) % row_size
            col_id = (tf.round(tf.cast(col_id, dtype=self.dtype_tf_float) + col) + col_size) % col_size

            selectorsTensor = tf.cast(tf.reshape(row_id * row_size + col_id, shape=(N*h*w,)), dtype=self.dtype_tf_int)
            #print("post somSigma selectorsTensor: ", selectorsTensor)

        ''' Sampling mask: zero selected sampling prototypes because sampling_placeholder entries were < 0 '''
        sampling_mask = tf.expand_dims(tf.cast(tf.greater(tf.reduce_sum(self.sampling_placeholder, axis=3), 0.), self.dtype_tf_float), 3)   # -> (N,hIn,wIn,1); only applied for pooling
        '''
        From this point on we have a "post-processed" topdown tensor from a higher hierachy layer to select mus/sigmas accordingly
        Need to distinguish between two cases:
            - conv_mode -> shared set of mus/sigmas for all receptive fields
            - ind mode  -> separate mus/sigmas for all receptive fields
        selectorsTensor: contains indices of centroids: 0,...,K
        Select either from:
            - mus[0,0,0], sigmas[0,0,0]     -> since there is only a single set of mus in conv_mode
            - or from all independent mus   -> in which case we have to do some unpleasant stuff (memory-wise)
        '''
        if self.conv_mode == 'yes':                                                                                     # for each patch/RF (filter), the whole input is one set of mus/sigmas
            selectedMeansTensor     = tf.gather(self.mus[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)               # select only the prototypes -> (N,D,); simply fancy indexing (gather)
            selectedSigmasTensor    = tf.gather(self.sigmas[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)            # selectorsTensor: indices for minibatch, copy selected components for mus & sigmas -> N, ?
        else:
            #FIXME: this is a horribly memory-inefficient way to select prototypes and sigmas, not very bad since mem is freed immediately afterwards, but if we should have time ... fix it!
            musTmp                  = tf.reshape(tf.stack([self.mus     for i in range(0, N)], axis=0), (N * h * w, self.K, -1))    # .--.-> Nhw,K,D
            sigmasTmp               = tf.reshape(tf.stack([self.sigmas  for i in range(0, N)], axis=0), (N * h * w, self.K, -1))    # .--.-> Nhw,K,D
            selectedMeansTensor     = tf.gather(musTmp, selectorsTensor, axis=1, batch_dims=1)                                      # select only the prototypes --> N , D
            selectedSigmasTensor    = tf.gather(sigmasTmp, selectorsTensor, axis=1, batch_dims=1)                                   # --> N, ?

        if self.mode == Mode.DIAG:                                                                                      # diag mode: covariance matrices only have diagonal entries (zero on non-diagonal entries)
            sigma_limit         = math.sqrt(self.sigma_upper_bound)
            mask                = tf.cast(tf.less(selectedSigmasTensor, sigma_limit), self.dtype_tf_float)              # clip sigma values
            covariancesTensor   = ((1. / (selectedSigmasTensor + 0.00001))) * mask                                      # convert from precision matrix to variances
            mean                = tf.cast(tf.reshape(selectedMeansTensor, (-1,)), self.dtype_tf_float)
            stddev              = tf.cast(tf.reshape(covariancesTensor, (-1,)) / self.sampling_divisor, self.dtype_tf_float) # sampling_divisor = 1: take covariances as is, or > 1: less noisy due to covariances getting smaller
            shape               = [N * w * h * cIn]
            mvn_tensor          = tf.random.normal(                                                                     # !!! this is where the magic happens !!!
                                    shape=shape,                                                                        # for 1 prototype K, 'pis' create some gaussian noise:
                                    mean=mean,                                                                          # picked from 'mus' (a single centroid point with 784 dims)
                                    stddev=stddev,                                                                      # and 'sigmas' (per pixel noise if the cov-matrix is diagonal)
                                    dtype=self.dtype_tf_float
            )
        sampling_op = tf.reshape(mvn_tensor, (N, h, w, cIn))
        '''
        - sampling_mask indicates where topdown was uniformly negative
        - indicating that a pooling layer in sparse mode sampled a column that is to be ignored
        - where mask (h,w,0) == 0, we set a whole channel to 0 this is sth the folding layer below can detect
        '''
        return sampling_op * sampling_mask


    def compute_output_shape(self, input_shape):
        ''' Returns a tuple containing the output shape of this layers computation. '''
        return self.batch_size, self.h_out, self.w_out, self.c_out


    def set_parameters(self, **kwargs):
        pass


    def get_layer_opt(self):
        ''' Returns the optimizer that is attached to this layer by the tfa.MultiOptimizer. '''
        ''' 
        FIXME: Exponential Decay for Opt.
        expdecay_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.eps_inf,
            decay_steps=20,
            decay_rate=.1)
        '''
        layer_opt = tf.keras.optimizers.SGD(learning_rate=self.tf_eps, momentum=self.sgd_momentum)

        return layer_opt


    def get_grad_factors(self):
        return {   
            self.mus.name      : self.lambda_mu_factor,
            self.sigmas.name   : self.lambda_sigma_factor,
            self.pis.name      : self.lambda_pi_factor
        }


    def build_layer_metrics(self):
        self.layer_metrics = [
            tf.keras.metrics.Mean(name=f'{self.prefix}loss')
        ]


    def get_layer_metrics(self):
        return self.layer_metrics


    def get_logging_params(self):
        return { 
            f'{self.prefix}K'                   : self.K,
            f'{self.prefix}reset_factor'        : self.reset_factor,
            f'{self.prefix}eps_0'               : self.eps_0,
            f'{self.prefix}eps_inf'             : self.eps_inf,
            f'{self.prefix}somSigma_0'          : self.somSigma_0,
            f'{self.prefix}somSigma_inf'        : self.somSigma_inf,
            f'{self.prefix}loss_masking'        : self.loss_masking,
            f'{self.prefix}sampling_divisor'    : self.sampling_divisor,
            f'{self.prefix}sampling_S'          : self.sampling_S,
            f'{self.prefix}sampling_P'          : self.sampling_P,
            f'{self.prefix}wait_threshold'      : self.wait_threshold,
            f'{self.prefix}wait_target'         : self.wait_target
        }
