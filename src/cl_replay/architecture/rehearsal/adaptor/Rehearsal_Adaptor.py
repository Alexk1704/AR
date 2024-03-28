import numpy as np
import tensorflow as tf

from cl_replay.api.experiment.adaptor import Supervised_Replay_Adaptor
from cl_replay.api.utils              import log



class Rehearsal_Adaptor(Supervised_Replay_Adaptor):


    def __init__(self, **kwargs):
        Supervised_Replay_Adaptor.__init__(self, **kwargs)
        
        self.storage_method         = self.parser.add_argument('--storage_method',      type=str,   default='reservoir', choices=['vanilla', 'reservoir'])
        self.storage_budget         = self.parser.add_argument('--storage_budget',      type=float, default=.05, help='use X perc. of total training data or fixed amount for storage buffer')
        self.budget_method          = self.parser.add_argument('--budget_method',       type=str,   default='class', choices=['class', 'task'])
        self.per_class_budget       = self.parser.add_argument('--per_class_budget',    type=int,   default=50)      
        self.per_task_budget        = self.parser.add_argument('--per_task_budget',     type=float, default=.001, help='if x < 0: percentage if x > 0: fixed amount')
        self.samples_to_generate    = self.parser.add_argument('--samples_to_generate', type=float, default=1., required=False, help='total amount or factor of generated samples for each replay sub-task')


    def sample(self, task, data=None, gen_classes=None, real_classes=None, **kwargs):
        if task > 1:    # const
            if self.samples_to_generate == -1.:
                stg = data.shape[0]
            else:       # balanced
                stg = data.shape[0] * len(gen_classes)
        return self.generator.sample_from_buffer(task=task, stg=stg, sbs=self.sampling_batch_size)


    def store(self, task, task_classes, task_data):
        if self.budget_method == 'class': # use a fixed amount of samples for each class
            amount_to_save = int(len(task_classes) * self.per_class_budget)
        else:
            if self.per_task_budget < 1.: # percentage
                amount_to_save = int(task_data[0].shape[0] * self.per_task_budget)-1
            else: # fixed amount
                amount_to_save = int(self.per_task_budget)

        self.generator.save_to_buffer(task, task_data, amount_to_save, self.storage_method)


    def before_subtask(self, task, total_samples=None, **kwargs):
        if task == 1:
            if self.storage_budget < 1.:
                self.storage_budget = int(total_samples * self.storage_budget)
            else:
                self.storage_budget = int(self.storage_budget)
            self.generator.init_buffers(self.storage_budget)


    def after_subtask(self, task, task_classes, task_data, **kwargs):
        self.store(task, task_classes, task_data)
        
    
    def set_class_freq(self, class_freq):
        self.class_freq = class_freq # compat
