import numpy as np
import tensorflow as tf

from cl_replay.api.utils import log
from cl_replay.api.data.Dataset import visualize_data


class DGR_Generator:


    def __init__(self, model, data_dims):
        self.model = model
        self.h, self.w, self.c, self.num_classes = data_dims


    def generate_data(self, task=-1, xs=None, gen_classes=None, stg=1000, sbs=100, vis_gen='no'):
        ''' generates samples from the old scholar (generator) '''
        log.debug('{:11s}'.format(' GENERATOR').center(128, '-'))
        
        if self.model.generator_type == 'VAE':
            log.debug(f'GEN_CLASSES: {gen_classes}, ENC. COND. INPUT: {self.model.enc_cond_input}, DEC. COND. INPUT: {self.model.dec_cond_input}')

        generated_data = []
        generated_labels = []

        for iteration in range(0, int(stg) // int(sbs)):
            labels = None
            if self.model.generator_type == 'GAN':
                if self.model.conditional == 'yes':
                    labels = gen_classes
                    rnd_ys = np.random.choice(labels, size=sbs)
                    tmp = tf.eye(sbs, self.num_classes)
                    ys = tf.gather(tmp, rnd_ys)
                else: ys = None
                gen_xs = self.model.generator.sample(sbs, ys)
                gen_xs = tf.reshape(gen_xs, (sbs, self.h, self.w, self.c))
                
            elif self.model.generator_type == 'VAE':
                if self.model.enc_cond_input == 'yes' or self.model.dec_cond_input == 'yes':
                    labels = gen_classes
                gen_xs = self.model.generator.sample(eps=None, batch_size=sbs, scalar_classes=labels)
            
            tmp_ys = self.model.solver.predict(gen_xs, verbose=0)
            amaxes = tmp_ys.argmax(axis=1)
            gen_xs = gen_xs.numpy()
            gen_ys = np.zeros(tmp_ys.shape)
            gen_ys[range(0,sbs), amaxes] = 1
            
            if vis_gen == 'yes' and iteration == 0:
                sort_indices = amaxes.argsort()
                gen_ys = gen_ys[sort_indices]
                gen_xs = gen_xs[sort_indices]
                visualize_data(gen_xs, gen_ys, self.model.vis_path, f'gen_T{task}')
            
            generated_data.append(gen_xs)
            generated_labels.append(gen_ys)
        concat_data = np.concatenate(generated_data, axis=0)
        concat_labels = np.concatenate(generated_labels, axis=0)
        
        #unique, counts = np.unique(concat_labels, return_counts=True)
        log.debug('{:8s}'.format(' LABELS').center(128, '-'))
        log.debug(f'GENERATED SAMPLES (CLASSES) COMPLETE DS: {concat_labels.sum()}, {concat_labels.sum(axis=0)}')
        log.debug('{:5s}'.format(' END').center(128, '-'))
        
        return concat_data, concat_labels
