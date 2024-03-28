import random
import numpy        as np
import tensorflow   as tf

from cl_replay.api.utils import log


class Sampler(object):
    ''' 
    Manages several data pairs (xs,ys) into sub-task "partitions" & yields mini-batches with certain proportions for stored pairs.
        - Meant to facilitate replay, where several datasets are generated from past sub-task.
        - Generated data is merged with novel data instances, and is sampled from this generator structure.
        - Sampling from each (x,y) pair is based strictly on the specified proportions f/e sub-task.
        - This means that the size of the x,y pairs is immaterial, and it is not guaranteed, that all samples from a sub-task will actually be processed.
        - This class can be used as a python-generator via yield(), by simply calling it as an iterator to generate data batch-wise (e.g. in model.fit).
        - You can also simply use a non-iterator interface by calling next_batch() to obtain a merged mini-batch
    '''
    
    def __init__(self, batch_size, dtype_np=np.float32, dtype_tf=tf.float32, real_sample_coef=1., gen_sample_coef=1.):
        self.batch_size         = batch_size
        self.sample_size        = None
        self.drop_remainder     = True
        self.dtype_np_float     = dtype_np
        self.dtype_tf_float     = dtype_tf
        self.real_sample_coef   = real_sample_coef
        self.gen_sample_coef    = gen_sample_coef

        self.reset()


    def reset(self):
        self.subtask_data           = []
        self.subtask_indices        = []
        self.raw_subtask_indices    = []
        self.subtask_batch_counters = []
        self.nr_samples             = 0
        self.nr_subtasks            = 0


    def add_subtask(self, xs, ys=None, index=-1):
        ''' x and y are numpy arrays '''
        self.nr_subtasks += 1
        if self.sample_size is None:
            self.sample_size = (xs.shape[1], xs.shape[2], xs.shape[3])
            if type(ys) is np.ndarray:
                self.no_labels = False
                self.label_size = ys.shape[1]
            else:
                if ys == []: self.no_labels = True

        self.raw_subtask_indices.append(np.arange(0, xs.shape[0]))
        np.random.shuffle(self.raw_subtask_indices[-1])

        self.subtask_indices.append(None)
        self.subtask_data.append((xs, ys))

        self.nr_samples             += xs.shape[0]
        self.subtask_batch_counters = [0] * self.nr_subtasks
        self.nr_subtask_batches     = [0] * self.nr_subtasks


    def replace_subtask_data(self, subtask_index, x, y):
        ''' Replace the data of one subtask by new data. 
            All existing structures are untouched. 
            Assumes that data shapes all conincide (all numPy here).
        '''
        self.subtask_data[subtask_index][0][:]        = x
        self.subtask_data[subtask_index][1][:]        = y
        self.subtask_batch_counters[subtask_index]    = 0


    def set_proportions(self, prop):
        ''' Sets the sub-task proportions for sampling. Expects a list with a float entry for each sub-task. '''
        if len(prop) != self.nr_subtasks: self.prop = [1.] * self.nr_subtasks  # default

        prop_sum = sum(prop) # compute effective batch size for each sub-task
        self.subtask_batch_sizes = np.array([int(self.batch_size * p / prop_sum) for p in prop])  # calculates batch sizes based on sub-task proportions

        # correct for rounding: if sum of all batch sizes is less than self.batch_size: distribute difference randomly
        diff = self.batch_size - sum(self.subtask_batch_sizes)
        for i in range(0, diff):
            random_subtask = random.randint(0, self.nr_subtasks - 1)
            self.subtask_batch_sizes[random_subtask] += 1

        log.info(f'CONFIG SUBTASK BATCH SIZES BASED ON PROPORTIONS: {self.subtask_batch_sizes} (CORRECTED DIFF OF {diff} TO {sum(self.subtask_batch_sizes)})')

        for i, ((xs, ys), ind, bs) in enumerate(zip(self.subtask_data,
                                                    self.raw_subtask_indices,
                                                    self.subtask_batch_sizes)):
            nr_subtask_samples          = xs.shape[0]
            self.nr_subtask_batches[i]  = nr_subtask_samples // bs
            corrected_indices           = ind[0:bs * self.nr_subtask_batches[i]]
            self.subtask_indices[i]     = np.reshape(corrected_indices, (self.nr_subtask_batches[i], bs))


    def reset_batch_counters(self):
        for i, _ in enumerate(self.subtask_batch_counters): self.subtask_batch_counters[i] = 0


    def reshuffle_indices(self):
        for ind in self.subtask_indices: np.random.shuffle(ind.ravel())


    def next_batch(self):
        ''' Non-iterator interface: simply call next_batch to obtain a merged mini-batch. '''
        xs_shape = (self.batch_size,) + self.sample_size
        batch_xs = np.zeros(shape=xs_shape, dtype=self.dtype_np_float)
        if self.no_labels == True: batch_ys = []
        else: batch_ys = np.zeros([self.batch_size, self.label_size], dtype=self.dtype_np_float)
        sample_weight = np.ones(shape=xs.shape, dtype=self.dtype_np_float)

        batch_end = 0  # populates a mini-batch, loop over all sub-task data and draw a fraction of a mini-batch according to chosen proportions
        for i, ((xs, ys), ind, bs, batch_counter) in enumerate(zip(
                                                                self.subtask_data,
                                                                self.subtask_indices,
                                                                self.subtask_batch_sizes,
                                                                self.subtask_batch_counters)):
            if i == 0:  sample_weight[batch_end:batch_end + bs] = self.real_sample_coef
            else:       sample_weight[batch_end:batch_end + bs] = self.gen_sample_coef
            
            batch_xs[batch_end:batch_end + bs, :] = xs[ind[batch_counter]]
            if self.no_labels == False: batch_ys[batch_end:batch_end + bs, :] = ys[ind[batch_counter]]
            batch_end += bs

        for i, _ in enumerate(self.subtask_batch_counters): # increase each sub-task batch counter by 1 and cycle if necessary
            self.subtask_batch_counters[i] += 1
            if self.subtask_batch_counters[i] >= self.nr_subtask_batches[i]:
                self.subtask_batch_counters[i] = 0

        return { 'x': batch_xs, 'y': batch_ys, 'sample_weight': sample_weight }


    def __call__(self):
        while True:
            xs_shape = (self.batch_size,) + self.sample_size
            batch_xs = np.zeros(shape=xs_shape, dtype=self.dtype_np_float)
            if self.no_labels == True: batch_ys = []
            else: batch_ys = np.zeros([self.batch_size, self.label_size], dtype=self.dtype_np_float)
            sample_weight = np.ones(shape=self.batch_size, dtype=self.dtype_np_float)

            batch_end = 0  # loop over all sub-task data and draw a fraction of a mini-batch according to chosen proportions
            for i, ((xs, ys), ind, bs, batch_counter) in enumerate(zip(
                                                                    self.subtask_data,
                                                                    self.subtask_indices,
                                                                    self.subtask_batch_sizes,
                                                                    self.subtask_batch_counters)):
                if i == 0:  sample_weight[batch_end:batch_end + bs] = self.real_sample_coef
                else:       sample_weight[batch_end:batch_end + bs] = self.gen_sample_coef
                
                batch_xs[batch_end:batch_end + bs, :] = xs[ind[batch_counter]]
                if self.no_labels == False: batch_ys[batch_end:batch_end + bs, :] = ys[ind[batch_counter]]
                batch_end += bs

            for i, _ in enumerate(self.subtask_batch_counters): # increase each sub-task batch counter by 1 and cycle if necessary
                self.subtask_batch_counters[i] += 1
                if self.subtask_batch_counters[i] >= self.nr_subtask_batches[i]:
                    self.subtask_batch_counters[i] = 0
            #print(batch_ys.argmax(axis=1), '\n', sample_weight)
            #print("!!!",batch_xs.shape, batch_ys.shape)
            #print(batch_ys.argmax(axis=1))
            #print(sample_weight)
            yield (batch_xs, batch_ys, sample_weight)
