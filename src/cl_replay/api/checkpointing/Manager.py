import os, sys
import tensorflow as tf

from cl_replay.api.utils    import log
from cl_replay.api.parsing  import Kwarg_Parser


class Manager:
    '''  A manager supporting the saving/loading of training progress (model vars/weights) to the file system. '''

    def __init__(self, **kwargs):
        parser          = Kwarg_Parser(**kwargs)

        self.exp_id     = kwargs.get('exp_id', None)
        self.model_type = kwargs.get('model_type', None)
        self.ckpt_dir   = parser.add_argument('--ckpt_dir', type=str, required=True, help='directory for checkpoint files')
        self.load_ckpt_from = parser.add_argument('--load_ckpt_from', type=str, help='provide custom checkpoint file path (omit .ckpt).')
        if os.path.isabs(self.ckpt_dir) == False:
            log.error("--chkpt_dir MUST BE AN ABSOLUTE PATH!")
            sys.exit(0)
        self.ckpt_dir  = os.path.join(self.ckpt_dir, "checkpoints")
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.filename   = os.path.join(self.ckpt_dir, f'{self.exp_id}-{self.model_type.split(".")[-1].lower()}-{{}}.ckpt')

        self.load_task  = kwargs.get('load_task', 0)
        self.save_All   = kwargs.get('save_All', 'yes')

    def load_checkpoint(self, model, task = None, **kwargs):
        ''' Load a model configuration via the checkpoint manager. '''
        if task is None: task = int(self.load_task)
 
        if task <= 0           : return 0, model

        if self.load_ckpt_from:
            ckpt_file = self.load_ckpt_from + f'-{{}}.ckpt'
            ckpt_file = ckpt_file.format(task)
        else:
            ckpt_file = self.filename.format(task)
        try:
            log.debug(f'{tf.train.list_variables(ckpt_file)}')
            model.load_weights(ckpt_file)
            log.info(f'RESTORED MODEL: {model.name} FROM CHECKPOINT FILE "{ckpt_file}"...')
        except Exception as ex:
            log.error(f'A PROBLEM WAS ENCOUNTERED LOADING THE MODEL: {model.name} FROM CHECKPOINT FILE "{ckpt_file}": {ex}')
            self.load_task = 0
            raise ex

        return task, model

    def save_checkpoint(self, model, current_task, **kwargs):
        ''' Saves the current session state to the file system. '''
        if self.save_All == False: return

        try:
            chkpt_filename = self.filename.format(current_task)
            model.save_weights(chkpt_filename)
            self.model_name = model.name
            log.info(f'SAVED MODEL WEIGHTS OF "{self.model_name}" AFTER TASK T{current_task} TO FILE "{chkpt_filename}"')
        except Exception as ex:
            log.error(f'PROBLEM WAS ENCOUNTERED SAVING THE CHECKPOINT FILE FOR MODEL: {self.model_name} AFTER TASK T{current_task}...')
            raise ex
