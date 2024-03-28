import numpy as np
import random

from cl_replay.api.utils import log


class Rehearsal_Buffer:

    """ ER with reservoir sampling:
        *) Drawn samples from M are mixed with current data for each task
        *) Reservoir sampling is sued to populate the buffer structure! (see e.g. Continual learning with tiny episodic memories.)
        *) Memory budget M is set to 50 per class
        *) Replay is done 1:1 -> "The number of samples from the replay buffer is always fixed to the same amount as the incoming samples" 
        *) We perform oversampling
    """

    def __init__(self, data_dims):
        """
            total_budget: the total amount of (xs,ys) sample pairs the memory can hold
            per_task_budget: how many samples will be saved with each consecutive task
            TODO: per-task, per-batch balancing
        """
        self.dtype_np_float = np.float32
        self.dtype_np_int = np.int32
        self.H, self.W, self.C, self.num_classes = data_dims

        self.storage_budget = 0


    def init_buffers(self, storage_budget):
        self.storage_budget = storage_budget

        self.buffer_xs = np.zeros(
            shape=(storage_budget, self.H, self.W, self.C), dtype=np.float32)     # N,H,W,C
        self.buffer_ys = np.zeros(shape=(
            storage_budget, self.num_classes), dtype=np.float32)           # assume one-hot

        self.last_index = 0


    def save_to_buffer(self, task, task_data, amount_to_save, method='reservoir'):
        log.debug(
            f"CURRENT TASK DATA SHAPE: {task_data[0].shape}, amount_to_save: {amount_to_save}")
        log.debug(f'BUFFER STORAGE_BUDGET: {self.storage_budget}')

        if method == 'reservoir':  # reservoir sampling O(n) algo
            data_xs = np.zeros(shape=(amount_to_save, self.H,
                               self.W, self.C), dtype=np.float32)
            data_ys = np.zeros(
                shape=(amount_to_save, self.num_classes), dtype=np.float32)

            i = amount_to_save - 1
            n = task_data[0].shape[0]
            # init with first i elements
            data_xs = task_data[0][:amount_to_save]
            data_ys = task_data[1][:amount_to_save]

            while (i < n):
                j = random.randrange(i+1)
                if (j < amount_to_save):
                    data_xs[j] = task_data[0][i]
                    data_ys[j] = task_data[1][i]
                i += 1
        else:   # "random" selection
            add_indices = np.random.choice(
                task_data[0].shape[0]-1, size=amount_to_save)
            data_xs = task_data[0][add_indices]
            data_ys = task_data[1][add_indices]

        # start index is set to next free space
        start_index = self.last_index
        # TODO: check if storage_budget has enough room after T1, if not exit with error msg
        # amount of new samples exceeds the storage capacity
        if ((self.last_index + amount_to_save) > self.storage_budget):
            # gives us the last indices b4 storage runs full
            remainder = self.storage_budget - self.last_index
            log.debug(
                f"NOT ENOUGH FREE SPACE: can save {remainder} more samples, others are replaced!!!")

            # storage not completely full, fill with subset of new data
            if remainder > 0:
                self.buffer_xs[start_index:] = data_xs[:remainder]
                self.buffer_ys[start_index:] = data_ys[:remainder]

            # need to free that many indices from past tasks
            to_replace = amount_to_save - remainder
            # sample indices from an uniform distribution (new data from incoming task stay unaffected)
            indices_to_replace = np.random.choice(self.last_index)
            log.debug(
                f'REPLACING INDICES: {indices_to_replace} with new samples from current task!')

            # write remaining data on specific indices from dist.
            self.buffer_xs[indices_to_replace] = data_xs[remainder+1:]
            self.buffer_ys[indices_to_replace] = data_ys[remainder+1:]

            self.last_index += remainder                        # update last index
        else:                                                   # storage has enough capacity, np
            # shift last_index by amount to save
            self.last_index += amount_to_save
            log.debug(
                f"ENOUGH FREE SPACE: occupying space up until index {self.last_index}")
            log.debug(
                f'FILLING BUFFER FROM: start_index {start_index} to last_index {self.last_index}')
            self.buffer_xs[start_index:self.last_index] = data_xs
            self.buffer_ys[start_index:self.last_index] = data_ys

        # print(np.mean(self.buffer_xs[:self.last_index]))
        # print(np.mean(self.buffer_xs[self.last_index+1:]))
        log.debug(
            f"CURRENT BUFFER CLASS DISTRIBUTION: {np.sum(self.buffer_ys, axis=0)}")


    def sample_from_buffer(self, task, stg, sbs=100):
        """ random selection from buffer under constraint of constant-time """
        log.info('{:11s}'.format(' BUFFER').center(128, '-'))
        log.info(f'DRAWING SAMPLES FOR T1-T{task - 1}')

        drawn_samples = np.zeros(
            [stg, self.H, self.W, self.C], dtype=self.dtype_np_float)
        drawn_labels = np.zeros([stg, self.num_classes],
                                dtype=self.dtype_np_float)

        for itr in range(0, int(stg) // int(sbs)):
            indices_sample = np.random.choice(self.last_index, sbs)

            drawn_samples[sbs*itr:sbs*(itr+1)] = self.buffer_xs[indices_sample]
            drawn_labels[sbs*itr:sbs*(itr+1)] = self.buffer_ys[indices_sample]

        log.debug('{:8s}'.format(' LABELS').center(128, '-'))
        log.debug(f'SAMPLED (CLASSES): {drawn_labels.sum(axis=0)}')
        log.debug('{:5s}'.format(' END').center(128, '-'))

        return drawn_samples, drawn_labels
