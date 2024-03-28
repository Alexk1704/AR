import datetime

from importlib import import_module
from importlib.util import find_spec

from tensorflow             import keras
from cl_replay.api.utils    import log
from cl_replay.api.parsing  import Kwarg_Parser


class Manager:
    ''' Returns importable callback objects as a list to pass to model.fit(). '''

    def __init__(self, **kwargs):
        parser = Kwarg_Parser(**kwargs)

        self.callback_paths     = parser.add_argument('--callback_paths',   type=str, default=[],    help='list of callback paths to search for modules')
        self.train_callbacks    = parser.add_argument('--train_callbacks',  type=str, default=[],    help='list of callbacks to pass to fit()')
        self.eval_callbacks     = parser.add_argument('--eval_callbacks',   type=str, default=[],    help='list of callbacks to pass to evaluate()')
        self.global_callbacks   = parser.add_argument('--global_callbacks', type=str, default=[],    help='list of callbacks to pass to both fit() & evaluate()')

        self.exp_id             = kwargs.get('exp_id', None)

        self.train_callbacks    = self.train_callbacks   if type(self.train_callbacks) == type([]) else [self.train_callbacks]
        self.eval_callbacks     = self.eval_callbacks    if type(self.eval_callbacks) == type([]) else [self.eval_callbacks]
        self.global_callbacks   = self.global_callbacks  if type(self.global_callbacks) == type([]) else [self.global_callbacks]
        self.load_callbacks(**kwargs)


    def load_callbacks(self, **kwargs):
        self.train_cbs      = list()
        self.eval_cbs       = list()
        self.global_cbs     = list()

        self.train_cbs      += self.load_custom_cb_modules(self.train_callbacks, self.callback_paths, **kwargs)
        #self.train_cbs      += self.load_tensorboard(log_dir='./results/logs/fit' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), histogram_freq=1, profile_batch='50,60')
        self.eval_cbs       += self.load_custom_cb_modules(self.eval_callbacks, self.callback_paths, **kwargs)
        self.global_cbs     += self.load_custom_cb_modules(self.global_callbacks, self.callback_paths, **kwargs)

        self.train_cbs      += self.global_cbs
        self.eval_cbs       += self.global_cbs

        log.debug(f'TRAIN CALLBACKS:')
        for cb in self.train_cbs: log.debug(f'\t{cb}')
        log.debug(f'EVAL CALLBACKS:')
        for cb in self.eval_cbs: log.debug(f'\t{cb}')


    def get_callbacks(self):            return self.train_cbs, self.eval_cbs


    @staticmethod
    def load_csv_logger(**kwargs):      return [keras.callbacks.CSVLogger(**kwargs)]

    '''
    To use tensorboard for visualization, start the tensorboard dashboard:
        $ tensorboard --logdir ~/SCCL/src/cl_replay/results/logs/fit

    Upload data for a permanent access link:
        $ !tensorboard dev upload \
        --logdir logs/fit \
        --name "experiment 1337" \
        --description "wow look at this" \
        --one_shot

    Use the tensorboard profiler
        $ pip install -U tensorboard_plugin_profile
    '''
    
    @staticmethod
    def load_tensorboard(**kwargs):     return [keras.callbacks.TensorBoard(**kwargs)]

    @staticmethod
    def load_custom_cb_modules(cb_list, search_paths, **kwargs):
        callback_list = []
        if not isinstance(cb_list, list): cb_list = [cb_list]
        if not isinstance(search_paths, list): search_paths = [search_paths]

        for c in cb_list:
            module_name = f'{c}'
            found_mod_descriptor = f'cl_replay.api.callback'
            for s_p in search_paths:
                search = f'{s_p}.{c}'
                cb_spec = find_spec(search)
                found = cb_spec is not None
                if found: 
                    found_mod_descriptor = search
            try:
                mod = import_module(found_mod_descriptor)
                mod_obj = getattr(mod, module_name)(**kwargs) # parse kwargs for found callback module
                callback_list.append(mod_obj)
            except ImportError:
                log.debug(f'Something went wrong importing the callback module: {found_mod_descriptor}')

        return callback_list
