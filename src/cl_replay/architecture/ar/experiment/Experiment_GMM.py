import os
import sys
import itertools
import numpy as np

from importlib                  import import_module
from importlib.util             import find_spec

from cl_replay.api.experiment   import Experiment
from cl_replay.api.utils        import helper, log


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_GMM(Experiment):
    ''' Defines a GMM experiment. '''

    def _init_parser(self, **kwargs):
        Experiment._init_parser(self, **kwargs)

        self.model_type                     = self.parser.add_argument('--model_type',                  type=str,   default='DCGMM',        help='class name in model sub-dirrectory to instantiate')
        self.perform_inpainting             = self.parser.add_argument('--perform_inpainting',          default='no', choices=['no', 'yes'],help='switch inpainting on/off')
        self.perform_variant_generation     = self.parser.add_argument('--perform_variant_generation',  default='no', choices=['no', 'yes'],help='switch vargen on/off')
        self.perform_sampling               = self.parser.add_argument('--perform_sampling',            default='no', choices=['no', 'yes'],help='switch pre/post sampling on/off')

        self.variant_gmm_root               = self.parser.add_argument('--variant_gmm_root',            type=int,   default=-1,             help='generate variants from which gmm layer downwards?')
        self.cond_sampling_classes          = self.parser.add_argument('--cond_sampling_classes',       type=int,   default=None,           help='condsampling for which classes?')
        if type(self.cond_sampling_classes) is type(1): self.cond_sampling_classes = [self.cond_sampling_classes]

        self.sampling_layer                 = self.parser.add_argument('--sampling_layer',              type=int,   default=-1,             help='which layer is used to start sampling?')
        self.nr_sampling_batches            = self.parser.add_argument('--nr_sampling_batches',         type=int,   default=1,              help='how many rounds of sampling to perform??')


    def do_sampling(self, prefix="sampling_"):
        if self.perform_sampling == 'no': return
        if not os.path.exists(self.results_dir): os.makedirs(self.results_dir)

        for i in range(0, self.nr_sampling_batches):
            T = self.cond_sampling_classes
            log.debug(f'T: {T}, SAMPLING LAYER: {self.sampling_layer}')
            batch = None

            save_path = os.path.join(self.results_dir + "/" + self.exp_id + "_" + str(i))
            if T is not None and self.sampling_layer == len(self.model.layers) - 1:
                log.debug(f'TOP DOWN SHAPE: {topdown_signal.shape}')
                topdown_signal, one_hot = self.model.construct_topdown_for_classifier(self.num_classes, 0.95, T)
                batch = self.model.sample_one_batch(topdown=topdown_signal)
                np.save(os.path.join(save_path, "_labels.npy"), one_hot.numpy())
            elif T is not None and self.sampling_layer < len(self.model.layers):
                batch = self.model.sample_one_batch(topdown=None, last_layer_index=self.sampling_layer)
                topdown_signal, one_hot = self.model.construct_topdown_for_classifier(self.num_classes, 0.95, T)
                np.save(os.path.join(save_path, "_labels.npy"), one_hot.numpy())
            else:
                batch = self.model.sample_one_batch(topdown=None, last_layer_index=self.sampling_layer)
            np.save(os.path.join(save_path, "_samples.npy"), batch.numpy())
            self.model.save_npy_samples(prefix="sampling_" , sampled=batch)


    def create_model(self):
        ''' 
        Instantiate a functional keras DCGMM, builds layers from imported modules specified via bash file parameters "--LX_".
            - Layer and model string are meant to be modules, like a.b.c.Layer or originate from the api itself (cl_replay.api.layer.keras). 
        '''
        log.debug(f'INSTANTIATING MODEL OF TYPE "{self.model_type}"')
        model_layers = dict()
        model_input_index = self.parser.add_argument("--model_inputs", type=int, default = 0, help="layer index of model inputs")
        if type(model_input_index) == type(1): model_input_index = [model_input_index]
        model_output_index = self.parser.add_argument("--model_outputs", type=int, default = -1, help="layer index of model outputs")

        #-------------------------------------------- INIT LAYERS
        for i in itertools.count(start=0):  # instantiate DCGMM layers
            layer_prefix        = f'L{i}_'
            layer_type          = self.parser.add_argument(f"--L{i}", type=str, default=None, help="Layer type")
            if layer_type is None: break    # stop if type undefined
            layer_input         = self.parser.add_argument('--input_layer', type=int, prefix=layer_prefix, default = 10000, help="Layer indices of input layers")
            log.debug(f'\tCREATING LAYER OF TYPE "{layer_type}", INPUT COMING FROM "{layer_input}"...')
                
            try:  # functional model layer creation
                target = helper.target_ref(targets=layer_input, model_layers=model_layers)
                if target is not None: # not input layer
                    layer_class_name=layer_type.split(".")[-1]
                    layer_obj = getattr(import_module(layer_type), layer_class_name)(name=f"L{i}",prefix=layer_prefix,**self.flags)(target)
                else: # input Layer
                    layer_obj = getattr(import_module("cl_replay.api.layer.keras"), layer_type)(name=f"L{i}",prefix=layer_prefix, **self.flags)
                    if hasattr(layer_obj, 'create_obj'):  # if a layer exposes a tensor (e.g. Input), we create a layer object after instantiating the layer module
                        layer_obj = layer_obj.create_obj()

                last_layer_ref = layer_obj  # set last layer to model_output as a fallback
                model_layers.update({i: layer_obj})
            except Exception as ex:
                import traceback
                log.error(traceback.format_exc())
                log.error(f'ERROR WHILE LOADING LAYER ITEM WITH PREFIX "{layer_prefix}": {ex}.')

        model_inputs = helper.target_ref(model_input_index, model_layers)

        if model_output_index == -1: model_output_index = last_layer_ref
        model_outputs = helper.target_ref(model_output_index, model_layers)

        #-------------------------------------------- INSTANTIATE AND INIT MODEL
        try:
          model_module = import_module(self.model_type)
          model_class = getattr(model_module, self.model_type.split(".")[-1])
        except Exception as ex:
          log.error(f'ERROR WHILE LOADING MODEL {self.model_type}')
        
        model = model_class(inputs=model_inputs, outputs=model_outputs, **self.flags)
        model.compile(run_eagerly=True, optimizer=None)
        model.summary()
        return model


    def before_task(self, current_task, **kwargs):
        self.do_sampling()
        if current_task >= 1:
            self.model.reset()

            if self.perform_variant_generation == 'yes':
                xs, ys = next(self.training_sets[current_task - 1])

                self.model.save_npy_samples(prefix='data_', sampled=xs)
                self.model.save_npy_samples(prefix='variants_', sampled=self.model.do_variant_generation(xs, bu_limit_layer=self.variant_gmm_root))

            if self.perform_inpainting == 'yes':
                xs, ys = next(self.training_sets[current_task - 1])

                n, h, w, c              = xs.shape
                mask                    = np.ones([n, h, w, c])
                mask[:, :, w // 2:, :]  = 0

                variants    = self.model.do_variant_generation(xs * mask)
                result      = variants * (1. - mask) + xs * mask

                self.model.save_npy_samples(prefix='data_', sampled=xs*mask)
                self.model.save_npy_samples(prefix='variants_', sampled=result)


    def after_task(self, task, **kwargs):
        self.do_sampling()


if __name__ == '__main__':
    Experiment_GMM().run_experiment()
