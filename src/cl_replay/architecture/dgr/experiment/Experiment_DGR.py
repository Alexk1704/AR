import itertools
import numpy as np
import tensorflow as tf

from importlib                  import import_module
from cl_replay.api.utils        import helper, log
from cl_replay.api.experiment   import Experiment_Replay
from cl_replay.api.model        import Func_Model, DNN

from cl_replay.architecture.dgr.model           import DGR
from cl_replay.architecture.dgr.model.dgr_gen   import VAE
from cl_replay.architecture.dgr.adaptor         import Supervised_DGR_Adaptor
from cl_replay.architecture.dgr.generator       import DGR_Generator

from cl_replay.architecture.rehearsal.buffer    import Rehearsal_Buffer

class Experiment_DGR(Experiment_Replay):


    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor        = Supervised_DGR_Adaptor(**self.parser.kwargs)
        self.model_type 	= self.parser.add_argument('--model_type', type=str, default='DGR-VAE', choices=['DGR-VAE', 'DGR-GAN'], help='Which architecture to use for DGR?')

    #-------------------------------------------- MODEL CREATION & LOADING
    def create_model(self):
        ''' 
        Instantiate a functional keras DGR dual-architecture, builds layers from imported modules specified via bash file parameters "--XX_".
            - Layer and model string are meant to be modules, like a.b.c.Layer or originate from the api itself (cl_replay.api.layer.keras). 
            - DGR uses 3 networks, as of such, the single models are defined by using their distinct prefix.
                - EX_ : encoder network (VAE)
                - GX_ : generator network (GAN)
                - DX_ : decoder/discriminator network (VAE/GAN)
                - SX_ : solver network
        '''
        log.debug(f'INSTANTIATING MODEL OF TYPE "{self.model_type}"')
        
        if self.model_type == 'DGR-VAE':
            for net in ['E', 'D', 'S']:
                sub_model = self.create_submodel(prefix=net)
                # each sub_model defines a "functional block"
                if net == 'E': vae_encoder = sub_model
                if net == 'D': vae_decoder = sub_model
                if net == 'S': dgr_solver  = sub_model
            self.flags.update({'encoder': vae_encoder, 'decoder': vae_decoder, 'solver': dgr_solver})
        
        if self.model_type == 'DGR-GAN':
            for net in ['G', 'D', 'S']:
                sub_model = self.create_submodel(prefix=net)
                if net == 'G': gan_generator        = sub_model
                if net == 'D': gan_discriminator    = sub_model
                if net == 'S': gan_solver           = sub_model
            self.flags.update({'generator': gan_generator, 'discriminator': gan_discriminator, 'solver': gan_solver})
        
        dgr_model = DGR(**self.flags)
        return dgr_model
    

    def create_submodel(self, prefix):
        model_layers = dict()
        model_input_index = self.parser.add_argument(f'--{prefix}_model_inputs', type=int, default=0, help="layer index of model inputs")
        if type(model_input_index) == type(1): model_input_index = [model_input_index]
        model_output_index = self.parser.add_argument(f'--{prefix}_model_outputs', type=int, default=-1, help="layer index of model outputs")
        #-------------------------------------------- INIT LAYERS
        for i in itertools.count(start=0):  # instantiate model layers
            layer_prefix        = f'{prefix}{i}_'
            layer_type          = self.parser.add_argument(f'--{prefix}{i}', type=str, default=None, help="Layer type")
            if layer_type is None: break    # stop if type undefined
            layer_input         = self.parser.add_argument('--input_layer', type=int, prefix=layer_prefix, default = 10000, help="Layer indices of input layers")
            log.debug(f'\tCREATING LAYER OF TYPE "{layer_type}", INPUT COMING FROM "{layer_input}"...')
                
            try:  # functional model layer creation
                target = helper.target_ref(targets=layer_input, model_layers=model_layers)
                if target is not None: # not input layer
                    layer_class_name=layer_type.split(".")[-1]
                    layer_obj = getattr(import_module(layer_type), layer_class_name)(name=f"{prefix}{i}", prefix=layer_prefix, **self.flags)(target)
                else: # input Layer
                    layer_obj = getattr(import_module('cl_replay.api.layer.keras'), layer_type)(name=f"{prefix}{i}", prefix=layer_prefix, **self.flags)
                    if hasattr(layer_obj, 'create_obj'):  # if a layer exposes a tensor (e.g. Input), we create a layer object after instantiating the layer module
                        layer_obj = layer_obj.create_obj()
                # fallback
                last_layer_ref = layer_obj  
                last_layer_ref_index = i
                
                model_layers.update({i: layer_obj})
            except Exception as ex:
                import traceback
                log.error(traceback.format_exc())
                log.error(f'ERROR WHILE LOADING LAYER ITEM WITH PREFIX "{layer_prefix}": {ex}.')

        model_inputs = helper.target_ref(model_input_index, model_layers)
        if model_output_index == -1: model_output_index = last_layer_ref_index
        model_outputs = helper.target_ref(model_output_index, model_layers)

        #-------------------------------------------- INSTANTIATE AND INIT MODEL
        if prefix == 'E' or prefix == 'G' or prefix == 'D':
            model_prefix = 'VAE-' if self.model_type == 'DGR-VAE' else 'GAN-'
            model = Func_Model(inputs=model_inputs, outputs=model_outputs, name=f'{model_prefix}{prefix}', **self.flags)
        if prefix == 'S':
            model = DNN(inputs=model_inputs, outputs=model_outputs, name=f'DGR-solver', **self.flags)
    
        model.compile(run_eagerly=True)
        model.summary()
        
        return model


    def load_model(self):
        Experiment_Replay.load_model(self)

        self.adaptor.set_model(self.model)
        data_dims = self.model.input_size
        self.adaptor.set_input_dims(data_dims[0], data_dims[1], data_dims[2], self.model.num_classes)
        self.adaptor.set_generator(DGR_Generator(model=self.model, data_dims=self.adaptor.get_input_dims()))

    #-------------------------------------------- DATA HANDLING
    def feed_sampler(self, task, current_data):
        super().feed_sampler(task, current_data)


    def generate(self, task, data, gen_classes, real_classes, **kwargs):
        xs, _ = data
        return self.adaptor.generate(task, xs, gen_classes, real_classes, **kwargs)

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        super().before_task(task, **kwargs)
        self.adaptor.before_subtask(
            task,
            total_samples=self.samples_train_D_ALL
        )


    def after_task(self, task, **kwargs):
        super().after_task(task, **kwargs)
        self.adaptor.after_subtask(
            task, 
            task_classes=self.tasks[task],
            task_data=self.training_sets[task]
        )


if __name__ == '__main__':
    Experiment_DGR().run_experiment()
