import os
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.callbacks import Callback
 
from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log


class Log_Metrics(Callback):
    ''' Implements a metric logging callback to save the evaluation/test data into .csv files utilizing pandas dataframes. ''' 

    def __init__(self, **kwargs):
        super(Log_Metrics, self).__init__()

        parser = Kwarg_Parser(**kwargs)
        
        self.log_training       = parser.add_argument('--log_training',     type=str,       choices=['yes', 'no'], default='no')
        self.log_path           = parser.add_argument('--log_path',         type=str,       required=True)
        self.dump_after_train   = parser.add_argument('--dump_after_train', type=str,       choices=['yes', 'no'], default='no')
        # self.dump_after_test    = parser.add_argument('--dump_after_test',  type=str,       choices=['yes', 'no'], default='no')
        if os.path.isabs(self.log_path) == False: log.error("--log_path must be absolute!")
        self.log_path           = os.path.join(self.log_path, "metrics")
        if not os.path.exists(self.log_path): os.makedirs(self.log_path)
        log.debug(f'Logging metrics file to: {self.log_path}')
        self.exp_id             = kwargs.get('exp_id', None)
        
        self.current_task       = int(kwargs.get('load_task', 0))
        self.run_ok             = False
        self.test_metric_names, self.test_metric_values = [], []
        self.train_metric_names, self.train_metric_values = [], []
        self.batch_ctr          = 0
        self.custom_name        = ""
        self.append_once        = False


    def __del__(self):
        if self.run_ok == False: return
        self.dump_to_csv()


    def dump_to_csv(self, mode='test'): 
        if mode == 'test':
            # if not self.append_once:
            #     self.test_metric_names.extend(['num_tasks', 'num_metrics'])
            #     self.test_metric_values.extend([self.current_task, len(self.model.metrics)])
            #     self.append_once = True
            data = [np.array(self.test_metric_values)]
            cols = self.test_metric_names
        else:
            data = [np.array(self.train_metric_values)]
            cols = self.train_metric_names
        
        df_new = pd.DataFrame(columns=cols, data=data)

        self.fname = os.path.join(self.log_path, f"{self.exp_id}_{mode}.csv")

        if os.path.exists(self.fname): # join data
            df_exist = pd.read_csv(self.fname, index_col=0)
            df_concat = pd.concat([df_exist, df_new], axis=1)
            df_concat.to_csv(self.fname)
        else: # write new
            df_new.to_csv(self.fname)


    def on_train_begin(self, logs=None):
        self.current_task += 1
        self.train_metric_names, self.train_metric_values = [], []
        self.batch_ctr = 0
        self.custom_name = ""


    def on_batch_end(self, batch, logs=None):
        self.batch_ctr += 1


    def on_epoch_end(self, epoch, logs=None):
        if self.log_training == 'no': return
        
        # FIXME: bad practice; find a better way for generic metrics...
        if 'step_time' in self.model.metrics[-1].name:
            avg_step_time = self.model.metrics[-1].result().numpy()
            epoch_duration = avg_step_time * self.batch_ctr
            self.train_metric_names.extend(
                [f"train_T{self.current_task}-E{epoch}_{self.model.name}_duration"]
            )
            self.train_metric_values.extend(
                [epoch_duration]
            )
            all_metrics = self.model.metrics[:-1]
        else:
            all_metrics = self.model.metrics

        self.train_metric_names.extend([f"train_T{self.current_task}-E{epoch}_{self.model.name}_" + m.name for m in all_metrics])
        self.train_metric_values.extend([m.result().numpy() for m in all_metrics])
        
        self.batch_ctr = 0


    def on_train_end(self, logs=None):
        self.run_ok = True
        if self.dump_after_train == 'yes': self.dump_to_csv(mode='train')

 
    def on_test_begin(self, logs=None): 
        self.test_batch_ctr = 0
        self.meaned_test_metrics = {}


    def on_test_batch_end(self, batch, logs=None):
        self.test_batch_ctr += 1.
        
        if 'step_time' in self.model.metrics[-1].name:
            all_metrics = self.model.metrics[:-1]
        else:
            all_metrics = self.model.metrics
        
        for m in all_metrics:
            m_k = f"test_T{self.current_task}-{self.model.test_task}_{self.model.name}_" + m.name
            m_v = m.result().numpy()
            if m_k in self.meaned_test_metrics:
                self.meaned_test_metrics[m_k] += m_v
            else:
                self.meaned_test_metrics.update({m_k : m_v})


    def on_test_end(self, logs=None):
        self.run_ok = True
        for m_k, m_v in self.meaned_test_metrics.items():
            m_v /= self.test_batch_ctr
            self.test_metric_names.extend([m_k])
            self.test_metric_values.extend([m_v])

        # if self.dump_after_test == 'yes': self.dump_to_csv()
        