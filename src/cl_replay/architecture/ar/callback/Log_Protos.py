import math
import os
import numpy            as np
import pandas           as pd

from pathlib            import Path
from tensorflow         import keras
from keras.callbacks    import Callback

from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log



def rm_dir(path):
    dir_ = Path(path)
    for sub in dir_.iterdir():
        if sub.is_dir():
            rm_dir(sub)
        else:
            sub.unlink()
    dir_.rmdir()



class Log_Protos(Callback):
    ''' Save trainables as .npy files, gets called either on epoch or train end. '''

    def __init__(self, **kwargs):
        super(Log_Protos, self).__init__()

        parser = Kwarg_Parser(**kwargs)
        self.save_protos    = parser.add_argument('--save_protos', type=str, choices=['on_epoch', 'on_train_end'], default='on_train_end')
        self.log_connect    = parser.add_argument('--log_connect', type=str, choices=['yes', 'no'], default='no')
        self.log_each_n_protos = parser.add_argument('--log_each_n_protos', type=int, default=1)
        self.vis_path       = parser.add_argument('--vis_path', type=str, required=True)
        if os.path.isabs(self.vis_path) == False: log.error("--vis_path must be absolute!")
        self.exp_id         = kwargs.get('exp_id', None)
        
        self.test_task, self.train_task = 0, int(kwargs.get('load_task', 0))
        self.current_epoch  = 0
        self.test_batch     = 0
        self.layers_to_log  = []
        self.saved_protos   = []


    def __del__(self):
        self.save()


    def save(self):
        # info: need this for vis script, could get rid of task information here ofc!
        for t, e, vars in self.saved_protos:
            save_dir = self.vis_path + f'/{self.exp_id}_protos_T{t}/E{e}'
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            for vname, v in vars:
                fname = f'{save_dir}/{self.exp_id}_{vname}.npy'
                np.save(fname, v)


    def init_log_layers(self):
        for layer in self.model.layers:
            if hasattr(layer, 'is_layer_type'):
                if layer.is_layer_type('GMM_Layer'):
                    if layer.log_bmu_activity == True: self.layers_to_log.append(layer)

    #-------------------------------------------- START: CALLBACK FUNCTIONS
    #-------------------------- TRAIN
    def on_train_begin(self, logs=None):
        if len(self.layers_to_log) == 0: self.init_log_layers()           
        
        self.train_task += 1
        self.current_epoch = 0
        self.test_batch = 0
        
        if len(self.model.sampling_branch) > 1:
            last_layer, _    = self.model.find_layer_by_prefix(f'L{self.model.sampling_branch[0]}')
        else:
            last_layer = self.model.layers[-1]
        if last_layer.is_layer_type('Readout_Layer') and self.log_connect == 'yes':
            self.train_comp_connects    = dict()
            self.C                      = last_layer.channels_out

            self.init_connectivity_structs(mode='train')


    def on_train_end(self, logs=None):
        if self.save_protos != "on_train_end": return

        if self.log_connect == 'yes':
            self.save_connectivities(mode='train')
            self.print_connectivities(mode='train')     # uncomment for console-printing of connectivities
            self.reset_connectivities(mode='train')     # reset connectivities
        
        self.save_vars(self.current_epoch)


    def on_train_batch_end(self, batch, logs=None):
        if self.log_connect == 'yes':
            self.log_connectivities(mode='train')
            self.log_responsibilities(batch=batch)


    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1

        if self.save_protos != "on_epoch": return

        if self.log_connect == 'yes':
            self.save_connectivities(mode='train')
            self.print_connectivities(mode='train', epoch=epoch)    # uncomment for console-printing of connectivities
            self.reset_connectivities(mode='train')                 # reset connectivities

        if (epoch % self.log_each_n_protos) == 0:
            self.save_vars(epoch)

    #-------------------------- TEST
    def on_test_begin(self, logs=None):
        if len(self.layers_to_log) == 0: self.init_log_layers()

        if self.model.test_task:
            self.test_task = self.model.test_task
        else:
            self.test_task += 1

        if len(self.model.sampling_branch) > 1:
            last_layer, _    = self.model.find_layer_by_prefix(f'L{self.model.sampling_branch[0]}')
        else:
            last_layer = self.model.layers[-1]
        if last_layer.is_layer_type('Readout_Layer') and self.log_connect == 'yes':
            self.test_comp_connects = dict()
            self.C                  = last_layer.channels_out

            self.init_connectivity_structs(mode='test')


    def on_test_end(self, logs=None):
        if self.log_connect == 'yes' and self.model.test_task == 'DAll':
            self.save_connectivities(mode='test')   # save connectivities to FS
            self.print_connectivities(mode='test')  # uncomment for console-printing of connectivities
        self.reset_connectivities(mode='test')      # reset connectivities after test


    def on_test_batch_end(self, batch, logs=None):
        if self.log_connect == 'yes':
            self.log_connectivities(mode='test')
            self.log_responsibilities(batch=batch)
    #-------------------------------------------- END: CALLBACK FUNCTIONS

    def log_connectivities(self, mode='train'):
        ''' 
        Backtracks GMM component activations and builds a connectivity mapping between classification and BMUs.
            - We use the output probabilities (responsibilities) of each GMM layer and connect them with the computed logits.
            - This allows to create a view of accumulated component activations across all local receptive fields or the global image. 
        '''

        if self.log_connect == 'yes':
            if mode == 'train': connectivity_struct = self.train_comp_connects
            if mode == 'test':  connectivity_struct = self.test_comp_connects
            probs_to_logits = self.top_layer.logits.numpy()                                 # dims = [N,C], shows logit response f/e sample
            class_response  = probs_to_logits.argmax(axis=1, keepdims=True)                 # highest class responses f/e patch
            for layer, l_con in connectivity_struct.items():
                l_resp              = layer.resp.numpy()                                    # get GMMs component probabilities -> [N,H,W,K]
                n, h, w, self.K     = probs_to_logits.shape[0], layer.h_out, layer.w_out, layer.K

                patch_bmus          = np.argmax(l_resp, axis=3, keepdims=True)              # N,H,W (containing BMU activation per H,W)
                flat_bmus           = patch_bmus.reshape((n,h*w))
                '''
                unique_bmu_picks    = np.zeros(shape=((n, K)))
                for i, per_sample_bmus in enumerate(flat_bmus):
                    u, c = np.unique(per_sample_bmus, return_counts=True)                   # count BMU picks for each sample
                    # "padding" for missing components (not selected)
                    re_c = np.zeros(shape=(K,), dtype=np.int32)
                    np.put_along_axis(re_c, u, c, axis=0)
                    unique_bmu_picks[i] = re_c
                print(f'\nUNIQUES (per sample):\n{unique_bmu_picks.shape}\n{unique_bmu_picks}\n')
                '''
                # add the distinct class counts at corresponding index while avoiding buffering problems
                np.add.at(l_con, (class_response, flat_bmus), 1)
                '''
                print(f'\nCLASS RESPONSE (ORIGINAL DIM):\n{class_response.shape}\n{class_response}')
                print(f'\nFLAT BMUs:\n{flat_bmus.shape}\n{flat_bmus}')
                print(f'\nFINAL CONNECTIVITY MAPPING:\n{l_con.sum()}\n{l_con}')
                '''


    def log_responsibilities(self, batch):
        if len(self.layers_to_log) > 0: ys = np.argmax(self.model.current_batch_ys, axis=1)
        else: return
        
        for layer in self.layers_to_log:
            n, C, h, w, K       = ys.shape[0], ys.max() + 1, layer.h_out, layer.w_out, layer.K
            gmm_resp            = layer.resp.numpy()
            gmm_resp            = gmm_resp.reshape((n,h*w, K))
            gmm_compactive      = np.argwhere(gmm_resp > 0.)
            class_labels        = ys[gmm_compactive[:,0]]
            gmm_compactive[:,0] = class_labels
            per_class_patch_act = np.zeros(shape=(C, h*w, K), dtype=np.int32)
            np.add.at(per_class_patch_act, (gmm_compactive[:,0], gmm_compactive[:,1], gmm_compactive[:,2]), 1)
            '''
            print(f'\n{layer.name} RESPONSIBILITIES (BATCH: {batch}):')
            for cls_id, data in enumerate(per_class_patch_act):
                if data.sum() != 0:
                    print(f'CLASS: {cls_id}')
                    print(f'{data[-1].reshape(int(math.sqrt(K)),int(math.sqrt(K)))}')
            '''


    def print_connectivities(self, mode='train', epoch=-1):
        if self.log_connect == 'no': return
        if mode == 'train':
            connectivities = self.train_comp_connects
            '''
            if epoch == -1: print(f'\nTRAIN CONNECTIVITIES (OVERALL):')
            else:           print(f'\nTRAIN CONNECTIVITIES (EPOCH: {epoch}):')
            '''
        if mode == 'test':
            connectivities = self.test_comp_connects
            #print(f'\nTEST CONNECTIVITIES (TASK {self.model.test_task}):\n')

        for layer_ref, comp_con in connectivities.items():
            K = int(math.sqrt(comp_con[0].shape[0]))
            col_ind = np.arange(start=0, step=1, stop=K)

            comp_sum = np.reshape(comp_con.sum(axis=0), [K, K])
            comp_ratio = np.around(np.reshape(comp_sum / comp_sum.sum(), [K, K]), decimals=4)
            '''
            print(F'\nLAYER:\t{layer_ref.name}\n')
            print(f'SUM OVER COMPONENTS:\n{comp_sum}\n\nACTIVATION RATIO:\n{comp_ratio}\n')
            print(f'GMM COMP. ACTIVATIONS PER CLASS\n')
            '''
            for i in range(0, comp_con.shape[0]):
                reshaped = np.reshape(comp_con[i], [K, K])
                class_comp_df = pd.DataFrame(
                    data=reshaped,
                    columns=col_ind,
                    index=col_ind
                )
                '''
                self.save_class_connectivities(class_label=i, data_frame=class_comp_df, mode=mode, epoch=epoch)
                print(f'\nCLASS {i}:\n {class_comp_df}')
                '''


    def save_connectivities(self, mode='train'):
        if self.log_connect == 'no': return

        if mode == 'train':
            connectivities = self.train_comp_connects
            fname = 'train_connectivities'
        if mode == 'test':
            connectivities = self.test_comp_connects
            fname = 'eval_connectivities'

        save_dir = self.vis_path + f'/protos/protos_T{self.train_task}/E{self.current_epoch}'

        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        for layer, l_con in connectivities.items():
            save_name = f'{save_dir}/{self.exp_id}_{layer.name}_{fname}.npy'
            np.save(save_name, l_con)


    def save_class_connectivities(self, class_label, data_frame, mode='train', epoch=-1, class_index=-1):
        if self.log_connect == 'no': return
        save_dir = self.vis_path + f'/class_connectivities_T{self.model.current_task}/'

        if mode == 'train': save_dir        += f'train/E{self.current_epoch}/'
        if mode == 'test':  save_dir        += f'test/E{self.current_epoch}/'
        fname           = f'C{class_label}.csv'
        fpath           = save_dir + fname

        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        data_frame.to_csv(fpath, index=False, header=False)


    def init_connectivity_structs(self, mode='train'):
        if self.log_connect == 'no': return
        self.top_layer = self.model.layers[-1]
        top_bottom_ref = self.model.layer_connectivity[self.top_layer.name]

        if mode == 'train': connectivity_struct = self.train_comp_connects 
        if mode == 'test':  connectivity_struct = self.test_comp_connects

        if hasattr(top_bottom_ref, 'is_layer_type'):
            if top_bottom_ref.is_layer_type('GMM_Layer'):
                sub_K               = top_bottom_ref.c_out
                sub_comp_connect    = np.zeros(shape=[self.C, sub_K], dtype=np.int32)
                connectivity_struct.update({top_bottom_ref: sub_comp_connect})
            if top_bottom_ref.is_layer_type('Concatenate_Layer'):
                top_bottom_subs         = self.model.layer_connectivity[top_bottom_ref.name]
                for sub_layer_ref in top_bottom_subs:
                    if sub_layer_ref.is_layer_type('Reshape_Layer'):
                        sub_sub_ref    = self.model.layer_connectivity[sub_layer_ref.name]
                        if not sub_sub_ref.is_layer_type('GMM_Layer'):
                            self.log_connect = False
                            raise Warning('THE SPECIFIED TOPOLOGY DOES NOT SUPPORT COMPONENT CONNECTIVITY LOGGING...')
                        else:
                            sub_K       = sub_sub_ref.c_out
                            sub_comp_connect = np.zeros(shape=[self.C, sub_K], dtype=np.int32)
                            connectivity_struct.update({sub_sub_ref: sub_comp_connect})


    def reset_connectivities(self, mode='train'):
        if self.log_connect == 'no': return
        if mode == 'train':
            for key, cons in self.train_comp_connects.items():
                con_shape = cons.shape
                self.train_comp_connects[key] = np.zeros(shape=cons.shape, dtype=np.int32)
        if mode == 'test' and self.log_connect:
            for key, cons in self.test_comp_connects.items():
                con_shape = cons.shape
                self.test_comp_connects[key] = np.zeros(shape=cons.shape, dtype=np.int32)


    def save_vars(self, epoch=1):
        # accumulate protos over epochs, only dump to storage when training ends!
        layer_vars = []
        for layer in self.model.layers:
            if layer.trainable:
                for v in layer.trainable_variables:
                    if v.name.find('GMM'):
                        vname = v.name.replace("/", "_")
                        if vname.find(":") != -1:
                            vname = vname.split(":")[0]
                        layer_vars.append((vname, v.numpy()))
        self.saved_protos.append((self.train_task, epoch, layer_vars))
