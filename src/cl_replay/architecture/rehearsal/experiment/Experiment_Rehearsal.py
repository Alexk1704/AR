import os
import sys
import math
import itertools
import numpy        as np
import tensorflow   as tf

from importlib      import import_module
from importlib.util import find_spec

from cl_replay.api.utils                        import log, helper
from cl_replay.api.experiment                   import Experiment_Replay
from cl_replay.api.model                        import DNN


from cl_replay.architecture.rehearsal.adaptor   import Rehearsal_Adaptor
from cl_replay.architecture.rehearsal.buffer    import Rehearsal_Buffer

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_Rehearsal(Experiment_Replay):
    """ Defines a basic experience replay experiment utilizing a buffer structure. """

    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor = Rehearsal_Adaptor(**self.parser.kwargs)

        self.model_type             = self.parser.add_argument('--model_type',      type=str, default='vanilla', choice=['vanilla', 'latent'], help='...')


    def _init_variables(self):
        Experiment_Replay._init_variables(self)

    #-------------------------------------------- MODEL CREATION/LOADING/SAVING
    def create_model(self):
        # TODO: see Experiment_DGR.py, allow layer creation via bash file...
        model_inputs  = tf.keras.Input(self.get_input_shape())
        if self.model_type == 'latent':
            flat    = tf.keras.layers.Flatten()(model_inputs)
            dense_1 = tf.keras.layers.Dense(512, activation="relu")(flat)
            dense_2 = tf.keras.layers.Dense(512, activation="relu")(dense_1)
            out     = tf.keras.layers.Dense(512, activation="relu")(dense_2)
        if self.model_type == 'vanilla':
            conv_1  = tf.keras.layers.Conv2D(32, (3, 3), (2, 2), padding="same", activation="relu")(model_inputs)
            pool_1  = tf.keras.layers.MaxPool2D((2, 2))(conv_1)
            conv_2  = tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding="same", activation="relu")(pool_1)  # INFO: alternative kernel_size (3, 3)
            pool_2  = tf.keras.layers.MaxPool2D((2, 2))(conv_2)
            flat    = tf.keras.layers.Flatten()(pool_2)
            dense_1 = tf.keras.layers.Dense(512, activation="relu")(flat)
            out     = tf.keras.layers.Dense(128, activation="relu")(dense_1)

        model_outputs  = tf.keras.layers.Dense(name="prediction", units=self.num_classes, activation="softmax")(out)

        model = DNN(inputs=model_inputs, outputs=model_outputs, **self.flags)
        model.compile(run_eagerly=True, optimizer=None)
        model.summary()
        return model


    def load_model(self):
        """ executed before training """
        super().load_model() # load or create model

        self.adaptor.set_input_dims(self.h, self.w, self.c, self.num_classes)
        self.adaptor.set_model(self.model)
        self.adaptor.set_generator(Rehearsal_Buffer(data_dims=self.adaptor.get_input_dims()))
        # setting callbacks manually is only needed when we train in "batch mode" instead of using keras' model.fit()
        if self.train_method == 'batch':
            for cb in self.train_callbacks: cb.set_model(self.model)
            for cb in self.eval_callbacks:  cb.set_model(self.model)
        else: return
            #TODO: add wandb support via keras callback


    def get_input_shape(self):
        return self.h, self.w, self.c

    #-------------------------------------------- DATA HANDLING
    def generate(self, task, data, gen_classes, real_classes, **kwargs):
        return self.adaptor.sample(task, data[0], gen_classes, real_classes, **kwargs)


    def replace_subtask_data(self, buffer_samples):
        """ replace subtask data (shapes have to coincide) of the sampler """
        x_buf, y_buf = buffer_samples
        self.sampler.replace_subtask_data(subtask_index=-1, x=x_buf, y=y_buf)
        log.debug(f'REPLACED SUBTASK DATA WITH BUFFER SAMPLES: {np.unique(np.argmax(y_buf, axis=1), return_counts=True)}')

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        """ draw N randomly selected training samples """	
        super().before_task(task, **kwargs)
        self.adaptor.before_subtask(
            task, 
            total_samples=self.samples_train_D_ALL
        )


    def after_task(self, task, **kwargs):
        """ select M samples from task data and save into buffer storage """
        super().after_task(task, **kwargs)
        self.adaptor.after_subtask(
            task, 
            task_classes=self.tasks[task],
            task_data=self.training_sets[task]
        )


if __name__ == '__main__':
    Experiment_Rehearsal().run_experiment()
