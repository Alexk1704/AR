import numpy            as np

from tensorflow         import keras
from keras.callbacks    import Callback

from cl_replay.api.utils.helper import wrap_in_arr



class Set_Model_Params(Callback):
    ''' Sets model parameters passed via **kwargs (external). '''

    def __init__(self, **kwargs):
        super(Set_Model_Params, self).__init__()
        self.l_observe = []


    def on_train_begin(self, logs=None):
        ''' Gather information about observed layer references. '''
        l_refs = []
        for il, l in enumerate(self.model.layers[1:], start=1):
            if hasattr(l, 'is_layer_type'):
                if l.is_layer_type('GMM_Layer') or l.is_layer_type('Readout_Layer'):
                    if l.wait_target[0] != None:
                        layer_targets = wrap_in_arr(l.wait_target)
                        layer_thresholds = wrap_in_arr(l.wait_threshold)
                        l_refs = []
                        for l_pre in layer_targets:
                            l_ref, _ = self.model.find_layer_by_prefix(l_pre)
                            if hasattr(l_ref, 'is_layer_type'):
                                if l_ref.is_layer_type('GMM_Layer'): l_refs.append(l_ref)
                        self.l_observe.append((self.model.layers[il], l_refs, layer_thresholds))
                        #l_refs.clear()


    def on_train_batch_begin(self, epoch, logs=None):
        ''' 
        Check layers and their targets if thresholds for observable tf_somSigma values have been reached. 
        Only allow training when threshold(s) reached, otherwise disable learning activity.
        '''
        for (obs_layer, l_refs, l_thresh) in self.l_observe:
            obs_layer.active = True 
            #print("obsl", obs_layer.name, l_refs)
            for i, l_t in enumerate(l_refs):
                #print("sigma of ", l_t.name, " = ", l_t.tf_somSigma.numpy()) ;
                if l_t.tf_somSigma.numpy() >= l_thresh[i]:
                    #print("Setting sigma of ", obs_layer, " top FALSE", l_t.tf_somSigma.numpy()) ;
                    obs_layer.active = False
