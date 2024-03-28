import os
import time
import itertools
import tensorflow                   as tf
import numpy                        as np

from tensorflow_addons.optimizers   import MultiOptimizer

from cl_replay.api.model            import Func_Model
from cl_replay.api.utils            import log, change_loglevel



class DCGMM(Func_Model):
    ''' 
        Defines a keras compatible (Deep Convolutational) Gaussian Mixture Model.

        Attributes
        ----------
        * vis_path : str, 
            - Sets a default dir to save data visualization.
        * batch_size : int, default=100
            - Size of fed training mini-batches.
        * TODO...
    '''
    def __init__(self, inputs, outputs, name="DCGMM", **kwargs):
        super(DCGMM, self).__init__(inputs, outputs, name, **kwargs)
        self.kwargs                 = kwargs

        self.vis_path               = self.parser.add_argument('--vis_path',            type=str,       required=True)
        self.batch_size             = self.parser.add_argument('--batch_size',          type=int,       default=100,            help='size of fed training mini-batches')
        #-------------------------- SAMPLING
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,       default=100,            help='size of mini-batches used for sampling (preferably the same as batch size)')
        self.sampling_divisor       = self.parser.add_argument('--sampling_divisor',    type=float,     default=1.,             help='divide std. devs in sampling by this factor')
        self.sampling_layer         = self.parser.add_argument('--sampling_layer',      type=int,       default=-1,             help='layer to sample from')
        self.sampling_branch        = self.parser.add_argument('--sampling_branch',     type=int,       default=[None],         help='specify the sampling order for backwards traversal of model layer')
        #-------------------------- TODO: OUTLIER DETECTION, UNUSED FOR NOW!
        self.outlier_layer          = self.parser.add_argument('--outlier_layer',       type=int,       default=-1,             help='log the losses for this layer on each test_step()')
        self.outlier_track_mode     = self.parser.add_argument('--outlier_track_mode',  type=str,       default='epoch',        choices=['epoch', 'step'], help='defines the interval of logging the loss of the outlier layer')
        #-------------------------- MASKING/LAYER MANIPULATION
        self.loss_masking           = self.parser.add_argument('--loss_masking',        type=str,       default='no',           choices=['no', 'yes'], help='turns on loss masking for the DCGMM')
        self.ro_layer_index         = self.parser.add_argument('--ro_layer_index',      type=int,       default=-1,             help='index of the layer in the hierarchy responsible for generating a mask based on inference')
        self.alpha_right            = self.parser.add_argument('--alpha_right',         type=float,     default=1.0,            help='lr for correctly classified samples')
        self.alpha_wrong            = self.parser.add_argument('--alpha_wrong',         type=float,     default=1.0,            help='lr for incorrectly classified samples')
        self.ro_patience            = self.parser.add_argument('--ro_patience',         type=int,       default=-1,             help='set to additionally train a Readout_Layer after GMM(s) convergence, -1 = no additional training; pos. int -> use patience (fixed epoch count of additional training)')
        self.set_default_opt        = self.parser.add_argument('--set_default_opt',     type=str,       default='yes',          choices=['no', 'yes'], help='if model should create its own optimizer instances')

        self.log_level              = self.parser.add_argument('--log_level',           type=str,       default='DEBUG',        choices=['DEBUG', 'INFO'], help='determine level for console logging.')
        change_loglevel(self.log_level)


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=True, steps_per_execution=None, **kwargs):
        self.opt_and_layers     = []    # tuple for MultiOptimizer
        self.layer_connectivity = { }   # e.g. { "L2_" : ["L2_", "L4_"], "L4_" : ["L6_"] }
        self.model_params       = { }
        self.all_metrics        = []

        self.sampling_layers    = None
        self.supports_chkpt     = True
        self.history_logs       = []
        self.current_task, self.test_task = 'T?', 'T?'

        log.debug(f'COMPILING DCGMM!')
        for i, layer in enumerate(self.layers[1:]):
            # create layer input connectivity mapping (dictionary containing keras layer names)
            if isinstance(layer.input_layer, int):
                l, _    = self.find_layer_by_prefix(f'L{layer.input_layer}')
                self.layer_connectivity.update({layer.name: l})

            if isinstance(layer.input_layer, list):
                ref_list = []
                for l in layer.input_layer:
                    l, _    = self.find_layer_by_prefix(f'L{l}')
                    ref_list.append(l)
                self.layer_connectivity.update({ layer.name : ref_list })

            if hasattr(layer, 'get_logging_params'): # collect important layer parameters for logging purpose
                self.model_params.update(layer.get_logging_params())

            if layer.trainable:
                try: # multiple layer optimizers, link each optimizer to a layer & append to list of tuples <optimizer, layer>
                    layer_opt = layer.get_layer_opt()   # get layer-specific optimizer
                    self.opt_and_layers.append((layer_opt, layer))
                    log.debug(f'\tCREATED LAYER OPTIMIZER: {layer_opt}')
                except Exception as ex: log.error(f'\tCOULD NOT OBTAIN VALID LAYER OPTIMIZER FOR {layer.prefix}: {ex}...')
                try: # metrics
                    layer.build_layer_metrics()        # build layer metrics (keras metrics) 
                    layer_metrics = layer.get_layer_metrics()
                    if layer_metrics != None: self.all_metrics.extend(layer_metrics)
                except Exception as ex: log.error(f'\tCOULD NOT OBTAIN VALID LAYER METRICS FOR {layer.prefix}: {ex}...')
        self.all_metrics.append( # run-time measurement
            tf.keras.metrics.Mean(name='step_time', dtype=self.dtype_tf_float))
        
        if self.set_default_opt == 'yes':
            if not optimizer:   self.opt = MultiOptimizer(self.opt_and_layers) # set optimizer for model
            else:               self.opt = optimizer
            log.debug(f'ADDED MODEL OPTIMIZER: {self.opt}')
        else: self.opt = None; log.debug(f'\tNO MODEL OPTIMIZER WAS SET...')
        # get ref to readout_layer if present, this is important for masking based on inference
        if self.ro_layer_index != -1: self.ro_layer, _    = self.find_layer_by_prefix(f'L{self.ro_layer_index}_')
        else:                         self.ro_layer       = None

        super(Func_Model, self).compile(optimizer=self.opt, run_eagerly=run_eagerly)


    def train_step(self, data, **kwargs):
        ''' Called by fit() & train_on_batch() and performs a single train-step (fwd & loss calculation) on a mini-batch of samples. '''
        #print(len(data), data[0].shape, data[1].shape)
        xs, ys = data[0], data[1]
        #np.savez(open("./debug.npz", "wb"), xs)
        self.current_batch_ys = ys

        self.pre_train_step()

        t1 = time.time()

        with tf.GradientTape(persistent=True) as tape:  # GT records forward pass of self.__call__()
            #-------------------------- FWD COMP
            self(inputs=xs, training=True)                   
            #-------------------------- CALC MASK
            with tape.stop_recording():                                        
                if self.loss_masking == 'yes': ro_mask = self.ro_layer.compute_mask_(ys) # calc inference-mask outside tape-context
            for i, (_, layer) in enumerate(self.opt_and_layers):
                #-------------------------- CALC LOSS
                raw_loss = layer.loss_fn(y_true=tf.stop_gradient(ys), y_pred=layer.get_fwd_result())
                #-------------------------- APPLY MASK
                if self.loss_masking == 'yes' and layer.get_masking_flag():
                    layer.apply_mask_to_raw_loss(ro_mask, self.alpha_right, self.alpha_wrong)
                #-------------------------- SET LOSS
                else:
                    layer.set_layer_loss(tf.reduce_mean(raw_loss) * -1.)
                #-------------------------- UPD METRICS
                with tape.stop_recording():
                    m = layer.get_layer_metrics()
                    if m is not None:
                        m[0].update_state(raw_loss)
                        for lm in ([] if len(m)<2 else layer.layer_metrics[1:]): #all layer_metrics >index 0 -> output + real ys
                            lm.update_state(ys, layer.get_output_result())
        #-------------------------- OPT LOSS            
        for opt, layer in self.opt_and_layers:
            _loss       = layer.get_layer_loss()
            _vars       = layer.trainable_variables
            _grads      = tape.gradient(_loss, _vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            _grads_vars = self.factor_gradients(zip(_grads, _vars), layer.get_grad_factors())
            opt.apply_gradients(_grads_vars)

        del tape

        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        self.all_metrics[-1].update_state(delta)

        self.post_train_step()

        return { m.name: m.result() for m in self.metrics }


    def test_step(self, data, **kwargs):
        xs, ys = data[0], data[1]
        self.current_batch_ys = ys
        
        t1 = time.time()
        
        self(inputs=xs, training=False)
        
        t2 = time.time()
        delta = (t2 - t1) * 1000.  # ms
        self.all_metrics[-1].update_state(delta)

        for (_, layer) in self.opt_and_layers:  # calculate losses for all trainable layers

            m = layer.get_layer_metrics()
            if m is not None:
                m[0].update_state(layer.loss_fn(y_true=ys, y_pred=layer.get_fwd_result()))
                if len(m) < 2: continue
                for lm in m[1:]:
                    lm.update_state(ys, layer.get_output_result())

        return { m.name: m.result() for m in self.metrics }


    def pre_train_step(self):
        for layer in self.layers[1:]: layer.pre_train_step()


    def post_train_step(self):
        for layer in self.layers[1:]: layer.post_train_step()


    def reset(self):
        for layer in self.layers[1:]: layer.reset_layer()


    def prepare_sampling(self):
        ''' Construct a hierarchical sampling structure for this model that is used in sampling mode. This is set via the --sampling_branch flag. '''
        self.sampling_layers = [None] * (len(self.sampling_branch) - 1)

        log.debug(f'SETTING UP SAMPLING HIERARCHY...')
        for i, j in zip(range(0, len(self.sampling_branch)-1), range(len(self.sampling_branch)-2, -1, -1)):
            layer_prefix        = self.sampling_branch[i]
            current_layer, _    = self.find_layer_by_prefix(prefix=f'L{layer_prefix}')
            if not current_layer:       raise Exception(f'\tLAYER "L{layer_prefix}" WAS NOT FOUND, PLEASE CHECK "--sampling_branch" FLAG...')
            if i < len(self.sampling_branch)-1:                 # do not set prev for last layer
                prev_id         = self.sampling_branch[i+1]     # get id of following elem
                prev_layer, _   = self.find_layer_by_prefix(prefix=f'L{prev_id}')
                if not prev_layer:      raise Exception(f'\tLAYER "L{layer_prefix}" WAS NOT FOUND, PLEASE CHECK "--sampling_branch" FLAG...')
                if isinstance(current_layer.input_layer, int):
                    current_layer.input_layer = [current_layer.input_layer]
                if prev_id in current_layer.input_layer:    # check connectivity
                    current_layer.prev_shape = prev_layer.output_shape
                    log.debug(f'\tSET SAMPLING PREDECESSOR FOR "{current_layer.name}" TO "{prev_layer.name}"')
                else:                   raise Exception(f'"{current_layer.name}" NOT CONNECTED TO "{prev_layer.name}", PLEASE CHECK "--sampling_branch" FLAG...')
            self.sampling_layers[j] = current_layer


    def sample_one_batch(self, topdown=None, last_layer_index=-1, **kwargs):
        '''
            Sample one batch from top to bottom (sampling branch). 
            Starts with layer specified via "last_layer_index" untill it reaches the lowest layer (input excluded).
            Performs sharpening if activated.

            Parameters
            ----------
            * topdown : tf.Variable
                - The output of the preceeding layer in the hierarchy.
            * last_layer_index : int, default=-1
                - Specifies the last layer in the sampling hierarchy.

            Returns
            -------
            * sampled : tf.Variable
                - The output of a backwards pass through this layer.

            ...
        '''
        if not self.sampling_layers:
            if len(self.sampling_branch) > 1:   self.prepare_sampling()                 # explicitly construct sampling order (e.g. for multi-headed models)
            if len(self.sampling_branch) <= 1:  self.sampling_layers = self.layers[1:]  # use default top-down layer structure

        if last_layer_index == -1: last_layer_index = len(self.sampling_layers) - 1
        last_layer  = self.sampling_layers[last_layer_index]
        sampled     = last_layer.backwards(topdown=topdown)
        sampled     = self.do_sharpening(last_layer, sampled)
        layer_index = last_layer_index - 1

        for layer in reversed(self.sampling_layers[0:layer_index+1]):
            sampled     = layer.backwards(topdown=sampled, **kwargs)
            log.debug(f'\tSAMPLING FROM: {layer.name} TOPDOWN TO LOWER: {sampled.shape}, MAX: {sampled.numpy().max()}')
            sampled     = self.do_sharpening(layer, sampled)
            layer_index -= 1

        return sampled


    def do_variant_generation(self, xs, selection_layer_index=-1):
        '''
            Performs variant generation for a given input tensor, i.e., performing a forward pass on given xs.
            This implicitly assumes that we have a top layer selected for the sampling procedure.

            Parameters
            ----------
            * xs : tf.Variable
                - Input tensor passed to the forward call of this model.
            * selection_layer_index : int, default=-1
                - Specifies the last layer in the sampling hierarchy.

            Returns
            -------
            * sampled : tf.Variable
                - The output of a backwards pass through this layer.

            ...
        '''
        layer_call_out          = self(xs)  # implicit assumption: lin. class is on top -> return logits
        selected                = self.layers[selection_layer_index]
        log.debug(f'\tSELECTED LAYER: {selected.name} WITH SHAPE: {selected.fwd.shape} FOR VARIANT GENERATION...')
        
        # INFO: simple check to investigate output logits in terms of generated classes
        # one_hot_fwd             = np.argmax(selected.fwd, axis=1)
        # unique, count           = np.unique(one_hot_fwd, return_counts=True)
        # log.debug(f'\tVAR LOGITS - CLASSES: {unique}, COUNT: {count}')
        
        return self.sample_one_batch(topdown=selected.fwd, last_layer_index=selection_layer_index)


    def do_sharpening(self, layer, X):
        '''
            Each Folding_Layer performs gradient ascent (GA) using the initial sampling result as a starting point.
            GA modifies the starting point to optimize the loss of the upstream target GMM_Layer.
            Usually, the targeted GMM should not be the direct successor, but one after that.

            Parameters
            ----------
            * X : tf.Variable
                - Input tensor passed from the upstream layer.
            * layer : keras.layers.Layer
                - Upstream layer

            Returns
            -------
            * varX : tf.Variable
                - The output of the sharpening iterations.
                - Or: Input tensor itself, if no sharpening was applied.

            ...
        '''
        target_layer = layer.get_target_layer()
        sharpening_chain = layer.get_sharpening_chain()
        if len(sharpening_chain) == 0: return X
        
        sharpening_rate         = layer.get_sharpening_rate()
        sharpening_iterations   = layer.get_sharpening_iterations()
        rec_weight              = layer.get_reconstruction_weight()

        curr_layer_id           = self.get_layer_index_by_name(layer.name)
        target_layer_id         = sharpening_chain[-1]
        target_layer,_          = self.find_layer_by_prefix(f"L{target_layer_id}_")
        last_layer_id           = len(self.layers)

        varX = tf.Variable(X, name="sharp",trainable=False)
        for i in range(0,sharpening_iterations):
            with tf.GradientTape() as g:
                output_ = varX # call layer to get output (resp for GMMMs)
                for layer_id in sharpening_chain:
                    #print(i, "traversing layer ", layer_id, "/", last_layer_id)
                    layer_      = self.find_layer_by_prefix(f"L{layer_id}_")[0]
                    output_     = layer_(output_, training=False)
                    #print (f"{layer_.name} out shape is ", output_.shape)
                fwd_        = target_layer.get_fwd_result() # access via instance attr since its computed in layer call
                target_layer_loss = target_layer.loss_fn(y_pred=fwd_)
                loss              = tf.reduce_mean(target_layer_loss - rec_weight * tf.reduce_mean((varX - X) ** 2))
            grad = g.gradient(loss, varX, unconnected_gradients='zero')
            varX.assign(varX + sharpening_rate * grad) ;
        return varX


    def construct_topdown_for_classifier(self, num_classes, maxconf, classes):
        '''
            Create an output control signal based on the desired classes.
            Setting the maximum confidence at the places of desired classes.

            Parameters
            ----------
            * num_classes: list
                - List of total classes.
            * classes : list
                - The classes to construct a topdown-signal for, e.g., [1,4,6]
            * maxconf : float
                - Max confidence for class 

            Returns
            -------
            * data : tuple
                - Returns a batch of sample-label tuples by randomly drawing samples based on the classes and their confidence 
                - Performs an inversion of the softmax.

            ...
        '''
        batch_size                      = self.sampling_batch_size
        minconf                         = (1. - maxconf) / (num_classes - 1)
        T                               = np.ones([batch_size, num_classes], dtype=self.dtype_np_float) * minconf
        one_hot                         = np.zeros([batch_size, num_classes], dtype=self.dtype_np_float)
        ax1                             = range(0, batch_size)

        rnd_drawn_samples               = np.random.choice(classes, size=batch_size)
        log.debug(f'\tCONSTRUCT TOPDOWN SIGNAL FOR: {classes}, DRAWN SAMPLES: \n{rnd_drawn_samples}')
        T[ax1, rnd_drawn_samples]       = maxconf
        one_hot[ax1, rnd_drawn_samples] = 1.0
        logT                            = tf.math.log(T) # inverse of SM

        return tf.constant(logT), tf.constant(one_hot)


    def save_npy_samples(self, sampled, prefix, mode="prototypes", sub_dir=None):
        '''
            Computes the DCGMM loss for some generated samples and saves them as pi's for visualization.

            Parameters
            ----------
            * sampled : tf.Variable
                - The output of a backwards pass through the model hierarchy.
            * prefix : str
                - A prefix to be added to the file descriptor.
            * sub_dir : str
                - A sub directory to be appended to "self.results_dir" 

            Returns
            -------
            * None

            ...
        '''
        N   = sampled.shape[0]
        d   = np.prod(sampled.shape[1:])
        if mode == 'prototypes':    sh = (1, 1, 1, N, d)
        else:                       sh = (N, d)
        loss = np.zeros([1, 1, 1, N])

        np.save(f'{self.vis_path}/{prefix}mus.npy', sampled.numpy().reshape(*sh))
        np.save(f'{self.vis_path}/{prefix}pis.npy', loss)
        np.save(f'{self.vis_path}/{prefix}sigmas.npy', np.zeros(sh))


    def set_parameters(self, **kwargs):
        '''
            Sets model/layer parameters by passing **kwargs, this is usually invoked by the experimental pipeline.

            Parameters
            ----------
            * **kwargs : dict
                - Containing various parameters targeted to the model or specific layers.

            Returns
            -------
            * None

            ...
        '''
        layer_kwargs = dict() # we filter **kwargs to only pass necessary params
        if 'sigma_state' in kwargs: # sigma_state is used to control the permission to train the Readout_Layer
            sigma_state = kwargs.get('sigma_state', 1.0)
            layer_kwargs.update({'sigma_state' : sigma_state})
        if 'class_freq' in kwargs:  # used for the calculation of the class frequencies, useful to perform a scaling based on class distribution
            class_freq = kwargs.get('class_freq')
            layer_kwargs.update({'class_freq' : class_freq})

        for l in self.layers[1:]:   # pass kwargs to model layers
            if hasattr(l, 'set_parameters'): l.set_parameters(**layer_kwargs)


    @property
    def metrics(self):
        ''' 
            Lists all "keras.metrics.Metric" objects from trainable layers.
            Metrics are appended to "self.all_metrics" attribute in self.compile().
            This executes self.reset_states() automatically at the end of each epoch.
        '''
        return self.all_metrics


    def get_model_params(self):
        ''' 
            Return a dictionary of model parameters to be tracked for an experimental evaluation via W&B. 
        '''
        return  { 
            'loss_masking' :        self.loss_masking,
            'loss_alpha_right' :    self.alpha_right,
            'loss_alpha_wrong' :    self.alpha_wrong,
            'ro_patience' :         self.ro_patience
        }

    @staticmethod
    def factor_gradients(grads_vars: zip, factors: dict) -> list:
        '''
            Multiply gradients of trainable variables with a preset factor.

            Parameters
            ----------
            * grads_vars : zip 
                - Resulting gradients recorded on the tape and corresponding trainable layer variables. 
            * factors : dict
                - Current factors to multiply gradients with.

            Returns
            -------
            * grads_vars : list
                - Returns gradients (multiplied with factors) and corresponding trainable layer variables.
            
            ...
        '''
        if len(factors) != 0: return [(g * factors[v.name], v) for (g, v) in grads_vars]
