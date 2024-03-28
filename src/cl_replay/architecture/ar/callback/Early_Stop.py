import numpy as np
import copy

from tensorflow             import keras
from keras.callbacks        import Callback
from cl_replay.api.utils    import log



class Early_Stop(Callback):
    ''' Implements early stoppage for the training procedure. '''

    def __init__(self, **kwargs):
        super(Early_Stop, self).__init__()
        self.patience       = int(kwargs.get('patience', 64))
        self.ro_patience    = kwargs.get('ro_patience', False)
        
        self.threshold      = 0.011
        self.converged      = False
        self.stop_at_epoch  = 0
        self.wait           = 0
        self.reason         = -1


    def on_train_begin(self, logs=None):
        self.stop_at_epoch          = 0
        self.reason                 = -1
        self.model.stop_training    = False

        self.wait                   = 0
        self.converged              = False

        if self.model.ro_patience > 0:
            self.ro_patience        = True
            self.ro_patience_epochs = self.model.ro_patience
        else:
            self.ro_patience        = False
            self.ro_patience_epochs = 0

        for layer in self.model.layers[1:]:
            if layer.is_layer_type('GMM_Layer'): layer.set_learning_rates(mu_factor=1.)  # (re-)activate learning for GMMs

        self.som_sigmas             = np.zeros(shape=self.model.layers.__len__(), dtype=np.float32)
        self.epoch_start_sigmas     = np.zeros(shape=self.model.layers.__len__(), dtype=np.float32)


    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_sigmas = copy.deepcopy(self.som_sigmas)


    def on_epoch_end(self, epoch, logs=None):
        # write all GMM somSigmas into structure
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'tf_somSigma'): self.som_sigmas[i] = layer.tf_somSigma.numpy()
        # get current max. somSigma
        self.highest_sigma = self.som_sigmas.max()
        # wait-counter
        start_end_eq = np.array_equiv(self.som_sigmas, self.epoch_start_sigmas)
        if start_end_eq:    self.wait += 1; log.debug(f'TF_SOM_SIGMA UNCHANGED, WAIT: {self.wait}')            # nothing changed from epoch start to epoch end
        else:               self.wait = 0; log.debug(f'TF_SOM_SIGMA CHANGED, RESETTING WAIT: {self.wait}')     # reset, if somSigmas changed
        
        # either ro_patience mode (wait until GMM converged) or standard patience mode (stationary training loss)
        if self.ro_patience:
            if (self.converged == True) and (self.wait >= self.ro_patience_epochs):
                self.model.stop_training    = True
                self.stop_at_epoch          = epoch
                self.reason                 = 0
        else:
            if self.wait >= self.patience:          # patience level has been reached (somSigma's stationary)
                self.model.stop_training    = True
                self.stop_at_epoch          = epoch
                self.reason                 = 0
        # either deactivate GMM training when converged (ro_patience mode) or stop training after GMM convergence
        if self.highest_sigma <= self.threshold:    # converged to threshold value
            if self.ro_patience == True and self.converged == False: # do not stop when GMM somSigma reached threshold (sigma0)
                for layer in self.model.layers[1:]:
                    if layer.is_layer_type('GMM_Layer'): layer.set_learning_rates(mu_factor=0.)   # deactivate learning for GMMs
                self.wait       = 0
                self.converged  = True
                log.debug(f'GMM(s) REACHED MIN. ANNEALING RADIUS... DISABLING GMM TRAINING')
            if not self.ro_patience:
                self.model.stop_training    = True
                self.stop_at_epoch          = epoch
                self.reason                 = 1


    def on_train_end(self, logs=None):
        if self.stop_at_epoch > 0:
            if self.reason == 0:
                log.info(F'STOPPED TRAINING AT EPOCH: {self.stop_at_epoch}, PATIENCE: {self.patience} / RO_PATIENCE: {self.ro_patience_epochs}... SOM_SIGMAS REMAINED STATIONARY...')
            if self.reason == 1:
                log.info(f'STOPPED TRAINING AT EPOCH: {self.stop_at_epoch}, THRESHOLD VALUE OF {self.threshold} REACHED FOR ALL SOM_SIGMAS...')
