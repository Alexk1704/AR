import numpy as np

from math import sqrt

from cl_replay.api.utils import log
from cl_replay.api.data.Dataset import visualize_data


class DCGMM_Generator:
    ''' 
        Externalize the DCGMM data generation procedure.

        Attributes
        ----------
        model : keras.models.Model
            - A model instance to generate data from.
        data_dims : tuple
            - A quadruple in the format of (H,W,C,N).
        dtype_float : np.dtype, optional
            - The data type of generated data.

        ...
    '''

    def __init__(self, model, data_dims, dtype_float=None, dtype_int=None):
        self.model = model
        if len(data_dims) == 4:
            self.h, self.w, self.c, self.num_classes = data_dims
        if len(data_dims) == 3:
            self.h, self.w, self.c = data_dims
            self.num_classes = None

        if dtype_float == None:  # if dtype not specified, try to infer it from model
            if hasattr(self.model, 'dtype_np_float'):
                self.dtype_np_float = self.model.dtype_np_float
            else:
                self.dtype_np_float = np.float32
        if dtype_int == None:
            if hasattr(self.model, 'dtype_np_int'):
                self.dtype_np_int = self.model.dtype_np_int
            else:
                self.dtype_np_int = np.int32


    def set_model(self, model):
        self.model = model


    def generate_data(
            self, task=-1, xs=None, gen_classes=None, generate_labels=True,
            stg=1000, sbs=100, sampling_layer=-1, sampling_clip_range=None,
            top_down=False, variants=False, vis_batch='no', vis_gen='no'):
        ''' 
            Generate data from the model instance.

            Parameters
            ----------
            task : int, optional, default=-1
                - Current task id.
            xs : np.array, tf.constant
                - Input data, only used for variant generation.
            gen_classes : list, optional 
                - The sub-set of past classes we want to generate samples for, only needed when conditionally sampling with a topdown-signal.
            stg : int, default=1000
                - Amount of samples to generate.
            sbs : int, default=100
                - Sampling batch size.
            sampling_layer : int, default=-1
                - Starting point of the backwards sampling transmission.
            sampling_clip_range : list, optional, default=None
                - Clips the generated samples to a range of [min - max].
            top_down : bool, default=False
                - Use GMM top-down sampling; requires a read-out layer for generating a topdown signal of logits for backwards transmission.
            variants : bool, default=False
                - Activate variant generation; requires xs as inputs to sample from corresponding BMU activations.
            generate_labels: bool, default = True
                - If true, labels are generated from xs data.

            Returns
            -------
            (gen_samples, gen_labels) : tuple
                - Returns the generated samples as a tuple of two numPy arrays.

            ...
        '''
        if top_down == True:
            try:
                gen_classes = np.array(gen_classes, dtype=self.dtype_np_int)
            except ValueError as ex:
                log.error(ex)
        else:
            if gen_classes == None:
                gen_classes = '?'

        if sampling_clip_range is None:
            sampling_clip_range = [0., 1.]

        log.debug('{:11s}'.format(' GENERATOR').center(128, '-'))
        log.debug(
            f'GEN_CLASSES: {gen_classes}\tTOP_DOWN: {top_down}\tVARIANTS: {variants}\tXS: {(xs.shape if xs is not None else xs)}')

        num_samples = int(stg)
        gen_samples = np.zeros(
            [num_samples, self.h, self.w, self.c], dtype=self.dtype_np_float)

        if generate_labels == True:
            gen_labels = np.zeros(
                [num_samples, self.num_classes], dtype=self.dtype_np_float)
        else:
            gen_labels = None

        # FIXME: compensate for 2 cases where incoming samples N < sbs OR data % sbs != 0 (remainder)
            # 1st case: we generate N samples, where N < sbs
            # 2nd case: cut data, omit generation of variants for remainder
        gen_range = num_samples // sbs
        for gen_it in range(0, gen_range):
            log.debug(
                f'ITER: {gen_it}, SBS: {sbs}, LOWER: {gen_it * sbs}, UPPER: {(gen_it + 1) * sbs}')
            # -------------------------- VARIANTS: generate from xs
            if variants and xs is not None:
                log.debug(f'GENERATING VARIANTS:')
                xs_batch = xs[gen_it * sbs:(gen_it + 1) * sbs]
                gen_xs = self.model.do_variant_generation(  xs_batch,
                                                            selection_layer_index=sampling_layer)
                if vis_batch == 'yes' and gen_it == 0:
                    visualize_data(xs_batch, None, f'{self.model.vis_path}/input', f'xs_T{task}')
            # -------------------------- TOPDOWN: generate from topdown signal
            elif top_down:
                log.debug(f'TOPDOWN SAMPLING:')
                topdown_logits, topdown_onehot = self.model.construct_topdown_for_classifier(   self.num_classes,
                                                                                                0.95, gen_classes)
                gen_xs = self.model.sample_one_batch(   topdown=topdown_logits,
                                                        last_layer_index=-1,
                                                        sampling_bs=sbs)
            # -------------------------- W/O TOPDOWN: sample w/o topdown signal
            else:
                log.debug(f'SAMPLING W/O TOPDOWN SIGNAL:')
                gen_xs = self.model.sample_one_batch(   topdown=None,
                                                        last_layer_index=sampling_layer,
                                                        sampling_bs=sbs)

            if vis_gen == 'yes' and gen_it == 0:
                visualize_data(gen_xs, None, f'{self.model.vis_path}/gen', f'gen_T{task}')

            # copy generated sample batch into generated dataset, and optionally stretch to interval
            clip_lo, clip_hi = sampling_clip_range
            npx = np.clip(gen_xs.numpy(), clip_lo, clip_hi)
            samplewise_max = npx.max(axis=(1, 2, 3))
            npx /= samplewise_max[:, np.newaxis, np.newaxis, np.newaxis]

            if generate_labels == True:
                # query model with generated samples
                generated_scalar_labels = self.model(npx).numpy()
                generated_scalar_labels = np.argmax(
                    generated_scalar_labels, axis=1)
                # log.debug(f'\nLABELS GATHERED FROM MODEL-QUERY:\n{generated_scalar_labels})
                # form one-hot representation of a sampled batch with dims (N,10)
                gen_ys = np.zeros([sbs, self.num_classes],
                                  dtype=self.dtype_np_float)
                gen_ys[range(0, sbs), generated_scalar_labels] = 1
                # concat labels to main data structure
                gen_labels[(gen_it * sbs):((gen_it + 1) * sbs)] = gen_ys

            gen_samples[gen_it * sbs:(gen_it + 1) * sbs] = npx

        if generate_labels == True:
            classes, counts = np.unique(
                gen_labels.argmax(axis=1), return_counts=True)
            total_cls = np.zeros(self.num_classes, dtype=np.float32)
            total_cls[classes] = counts
            log.debug('{:8s}'.format(' LABELS').center(128, '-'))
            log.debug(f'GENERATED {gen_labels.shape}: {total_cls}')
        log.debug('{:5s}'.format(' END').center(128, '-'))

        return gen_samples, gen_labels
