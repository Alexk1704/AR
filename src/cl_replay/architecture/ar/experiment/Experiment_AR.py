import os
import sys
import math
import itertools
import numpy       as np

from importlib import import_module
from importlib.util import find_spec

from cl_replay.api.utils                    import log, helper
from cl_replay.api.experiment               import Experiment_Replay
from cl_replay.architecture.ar.experiment   import Experiment_GMM
from cl_replay.architecture.ar.adaptor      import AR_Supervised
from cl_replay.architecture.ar.generator    import DCGMM_Generator


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_AR(Experiment_Replay):
    ''' Defines an experiment for AR. '''

    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor = AR_Supervised(**self.parser.kwargs)


    def _init_variables(self):
        Experiment_Replay._init_variables(self)
        self.adaptor.model_outputs = self.model_outputs

    #-------------------------------------------- MODEL CREATION & LOADING
    def create_model(self):
        return Experiment_GMM.Experiment_GMM.create_model(self)


    def load_model(self):
        ''' Loads a model state and lets all needed dependancies. '''
        super().load_model()
        self.adaptor.set_input_dims(self.h, self.w, self.c, self.num_classes)
        self.adaptor.set_model(self.model)
        self.adaptor.set_generator(DCGMM_Generator(model=self.model, data_dims=self.adaptor.get_input_dims()))
        # setting callbacks manually is only needed when we train in "batch mode" instead of using keras' model.fit()
        if self.train_method == 'batch':
            for cb in self.train_callbacks: cb.set_model(self.model)
            for cb in self.eval_callbacks:  cb.set_model(self.model)
        else: return #TODO: add wandb support via keras callback


    def get_input_shape(self):
        return self.h, self.w, self.c

    #-------------------------------------------- DATA HANDLING
    def generate(self, task, data, gen_classes, real_classes, **kwargs):
        xs, _ = data
        
        generate_labels = False
        if self.ml_paradigm == 'supervised': generate_labels = True
        
        return self.adaptor.generate(task, xs, gen_classes, real_classes, generate_labels, **kwargs)


    def replace_subtask_data(self, variants):
        ''' Replace the subtask data of the sampler. '''
        x_vars, y_vars = variants
        self.sampler.replace_subtask_data(subtask_index=-1, x=x_vars, y=y_vars)
        log.debug(f'REPLACED SUBTASK DATA WITH VARIANTS: {np.unique(np.argmax(y_vars, axis=1), return_counts=True)}')

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        self.adaptor.before_subtask(task)
        super().before_task(task, **kwargs)


    def after_task(self, task, **kwargs):
        super().after_task(task, **kwargs)
        self.adaptor.after_subtask(task)


if __name__ == '__main__':
    Experiment_AR().run_experiment()
