import numpy as np

from tensorflow         import keras
from keras.callbacks    import Callback



class Log_Loss(Callback):
    ''' Logs the losses for the selected outlier detection layer and saves them to the model structure. '''
    
    def __init__(self, **kwargs):
        super(Log_Loss, self).__init__()


    def on_epoch_end(self, epoch, logs=None):
        if self.model.outlier_track_mode == 'epoch':
            self.grab_outlier_log()


    def on_train_batch_end(self, batch, logs=None):
        if self.model.outlier_track_mode == 'step':
            self.grab_outlier_log()


    def grab_outlier_log(self):
        ''' Grabs the raw layer loss (per sample) of selected outlier det. layer and extends historic logs with meaned value. '''
        outlier_layer = self.model.outlier_layer
        out = self.model.layers[outlier_layer].get_raw_layer_loss()
        self.model.history_logs.append(np.mean(out[self.model.bool_mask]))