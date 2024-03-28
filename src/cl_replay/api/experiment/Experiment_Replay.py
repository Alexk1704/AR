import sys
import math
import wandb
import numpy       as np
import tensorflow  as tf

from cl_replay.api.data         import Sampler
from cl_replay.api.data.Dataset import visualize_data
from cl_replay.api.utils        import log, helper
from cl_replay.api.experiment   import Experiment


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_Replay(Experiment):
    ''' Defines an experimental pipeline performing replay. '''

    def _init_parser(self, **kwargs):
        Experiment._init_parser(self, **kwargs)


    def _init_variables(self):
        Experiment._init_variables(self)

        self.sampler            = Sampler(batch_size=self.batch_size)
        self.past_tasks         = 0
        self.generated_dataset  = None
        if self.single_class_test == 'yes': self.load_single_class_test()   

    #-------------------------------------------- DATA PROCESSING
    def feed_sampler(self, task, current_data):
        cur_xs, cur_ys = current_data
    
        current_task_iters = self.get_task_iterations(task)
        #-------------------------------------------- FIRST (INITIAL) TRAINING: SUBTASK 0
        if task == 1:
            self.sampler.add_subtask(xs=cur_xs, ys=cur_ys)
            self.sampler.set_proportions([1.])
            if self.ml_paradigm == 'supervised':
                class_ratio, self.class_freq = self.calc_class_freq(total_classes=self.DAll, targets=cur_ys, mode='ds')
                self.adaptor.set_class_freq(class_freq = self.class_freq)
            else: self.class_freq = None
            self.train_steps = current_task_iters[0]
        #-------------------------------------------- REPLAY TRAINING: SUBTASK >1 -> GENERATED DATA
        if task > 1:
            gen_xs, gen_ys = self.generated_dataset
            
            self.sampler.add_subtask(xs=cur_xs, ys=cur_ys)  # SUBTASK 0 -> current task real data
            self.sampler.add_subtask(xs=gen_xs, ys=gen_ys)  # SUBTASK 1 -> generated data

            if self.adaptor.replay_proportions == [-1., -1.] or not self.adaptor.replay_proportions:
                self.replay_proportions = np.array(self.calc_replay_props(
                    real_classes=self.tasks[task],
                    generated_classes=self.past_classes)
                ).astype(np.float32)
            else:
                self.replay_proportions = np.array(self.adaptor.replay_proportions).astype(np.float32)

            if self.ml_paradigm == 'supervised':
                merged_targets = np.concatenate((cur_ys, gen_ys))
                class_props, self.class_freq = self.calc_class_freq(total_classes=self.DAll, targets=merged_targets, mode='ds')

            self.sampler.set_proportions(self.replay_proportions)

            s = self.replay_proportions.sum()
            factor = s / self.replay_proportions[0]
            self.train_steps = math.ceil(current_task_iters[0] * factor)

            log.info(f'SETTING UP "steps_per_epoch"... ITERATIONS FOR CURRENT TASK: {current_task_iters}, '
                     f'ITERATIONS FOR MERGED TASK (CURRENT + REPLAYED): {self.train_steps}')

            log.info(f'\tADDED CURRENT TASK DATA FROM T{task} (TRAINING SET) TO THE REPLAY_SAMPLER...')

    @staticmethod
    def calc_replay_props(real_classes, generated_classes):
        ''' Calculate proportions dynamically based on class distribution (set bash-file params to: -1 -1 for auto mode). '''
        total_cls   = len(generated_classes) + len(real_classes)
        gen_prop    = int(len(generated_classes) / total_cls * 100)
        real_prop   = 100 - gen_prop

        return [real_prop, gen_prop]

    @staticmethod
    def calc_class_freq(total_classes, targets, mode='ds'):
        ''' Calculate the class freq. based on the distribution of target data. '''
        if mode == 'ds':
            class_props         = targets.sum(axis=0) / targets.sum()
            class_freq          = 1. - (class_props / 1.)
            return class_props, class_freq

        if mode == 'batch':
            classes, counts     = np.unique(targets, return_counts=True)
            total_cls           = np.zeros(shape=len(total_classes), dtype=np.float32)
            total_cls[classes]  = counts
            class_props         = total_cls / total_cls.sum()
            class_freq          = 1. - (class_props / 1.)
            return class_props, class_freq

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        ''' Prepare the model/exp. pipeline for the next sub-task. '''
        self.adaptor.model.current_task = task  # info about task (SL) for e.g. logging callbacks
        current_train_set = self.training_sets[task]
        self.past_tasks += 1
        
        self.past_classes, self.real_classes = [], []
        if self.ml_paradigm == 'supervised':                              
            for past_task in self.tasks[1:task]: self.past_classes.extend(past_task)
            self.real_classes = self.tasks[task]
        if task > 1:
            self.generated_dataset = self.generate(task, data=current_train_set, gen_classes=self.past_classes, real_classes=self.real_classes)
            self.sampler.reset()
            if self.loss_coef == 'class_balanced':
                past_cls = len(self.past_classes)
                current_cls = len(self.tasks[task])
                r_s_c = float(current_cls / (past_cls + current_cls)) 
                g_s_c = 1. - r_s_c
            elif self.loss_coef == 'task_balanced':
                r_s_c = float((1. / self.past_tasks))
                g_s_c = 1. - r_s_c
            elif self.loss_coef == 'off':
                r_s_c = 1.; g_s_c = 1.
            """
            else:
                if not len(self.gen_sample_coef) == (self.num_tasks - 1): # expecting list
                    log.error(f'SPECIFIED GEN. SAMPPLE COEF.: {self.gen_sample_coef} WHICH IS NOT MATCHING REPLAY TASK LENGTH, FALLBACK TO DEFAULT (1.).')
                    r_s_c = 1.; g_s_c = 1.
                else:
                    g_s_c = float(self.gen_sample_coef[task-2])
                    r_s_c = 1. - g_s_c
            """
            log.debug(f'USING THE FOLLOWING SAMPLE WEIGHTS: GEN. DATA -> {g_s_c} / REAL DATA -> {r_s_c}')
            self.sampler.real_sample_coef = r_s_c
            self.sampler.gen_sample_coef = g_s_c
        self.feed_sampler(task, current_train_set)


    def train_on_task(self, current_task):
        ''' Either use the automated keras .fit() or custom training pipeline using model.train_on_batch(). '''
        if self.train_method == 'batch':
            self.train_batch_mode(data=self.sampler(), epochs=self.epochs, train_steps=self.train_steps, current_task=current_task)
            return
        log.info('{:20s}'.format(f' [START] TASK: {current_task} TOTAL EPOCHS: {self.epochs} ').center(128, '-'))
        train_history = self.model.fit(self.sampler(),
                                       epochs=int(self.epochs[current_task-1]),
                                       batch_size=self.batch_size,
                                       steps_per_epoch=self.train_steps,
                                       callbacks=self.train_callbacks,
                                       verbose=self.verbosity)


    def train_batch_mode(self, data, epochs=100, train_steps=1000, current_task=None):
        ''' Defines a custom training loop. '''
        for cb in self.train_callbacks: cb.on_train_begin()
        global_iter = 0

        np.set_printoptions(linewidth=128) # FIXME adjust to dynamic class ratio printing
        for t_e in range(0, int(epochs[(current_task-1)])):
            #-------------------------- START EPOCH
            for cb in self.train_callbacks: cb.on_epoch_begin(epoch=t_e)
            log.info('{:20s}'.format(f' [START] TASK: {current_task} EPOCH: {t_e} ').center(128, '-'))

            for t_s in range(0, int(train_steps)):                          
                #-------------------------- START TRAINING ITERATION
                batch_data = data.__next__() # draw a batch from the data source iterator
                xs_data, targets = batch_data[0], batch_data[1]

                #-------------------------- LABELS
                if targets is not None:
                    target_classes = tf.math.argmax(targets, axis=1)
                    numpy_cls = target_classes.numpy()
                    ''' 
                    # INFO: counts classes of targets inside a np struct; can be accumulated over steps/epochs
                    unique, counts = np.unique(numpy_cls, return_counts=True)
                    re_c = np.zeros(shape=(len(self.DAll),), dtype=np.int32) # "padding" for missing classes
                    np.put_along_axis(re_c, unique, counts, axis=0)
                    '''
                    class_ratio, self.class_freq = self.calc_class_freq(total_classes=self.DAll, targets=numpy_cls, mode='batch')
                    # we set this here because "Set_Model_Params" callback triggers set_parameters() automatically before each batch
                    self.class_freq = np.around(self.class_freq, decimals=2)
                    self.model.class_freq = self.class_freq
                if self.num_classes > 10:   # set some numPy print options for console logging...
                    per_row_items   = (int(math.sqrt(self.num_classes)), int(math.sqrt(self.num_classes)))
                    width_per_entry = ((per_row_items[0]+1)*4) + (per_row_items[0]+2)
                    np.set_printoptions(linewidth=width_per_entry)

                #-------------------------- BEGIN TRAINING STEP
                for cb in self.train_callbacks: cb.on_train_batch_begin(global_iter)
                metric_results = self.model.train_on_batch(x=xs_data, y=targets, return_dict=True)
                for cb in self.train_callbacks: cb.on_train_batch_end(global_iter)

                #-------------------------- METRICS EVAL
                if t_s % 50 == 0:
                    log.debug(f'[TRAIN]\tSTEP\t{global_iter}')
                    if class_ratio is not None:
                        log.debug(f'\tCLASS RATIO:\t{class_ratio}')

                    if self.wandb_active == 'yes':
                        wandb_metrics_group = f't{current_task}_train'
                        wandb.log({wandb_metrics_group: metric_results}, step=global_iter)
                global_iter += 1
                #-------------------------- END TRAINING ITERATION

            #-------------------------- METRICS PRINT
            metric_str = ''
            for m_k, m_v in metric_results.items():
                metric_str += '{:16s} {:8.2f}'.format(m_k, metric_results[m_k]) + '\t'
            log.info(f'[TRAIN]\tSTEP\t{global_iter}:\t' + f'{metric_str}')
            
            for t_cb in self.train_callbacks: t_cb.on_epoch_end(epoch=t_e)
            #-------------------------- EPOCH END
            self.model.reset_metrics()  # call reset_metrics to clear accumulated layer metrics
            ''' 
            # INFO: comment in to allow per epoch testing!
            if t_e != (int(epochs) -1): # after each epoch (except last) -> eval via model.evaluate()
                self._test(current_task, global_iter)  
            '''
            if self.model.stop_training: break
            t_e += 1

        for t_cb in self.train_callbacks: t_cb.on_train_end()


    def after_task(self, task, **kwargs):
        pass


    def _test(self, current_task, current_step=None):
        ''' 
        Extends _test to perform evaluation for supervised training via keras model.evaluate or batch_wise.
            * if self.train_method is set to 'batch' -> batch_wise testing routine run with test_batch_mode()
            * if self.train_method is set to 'fit' -> model.evaluate() routine
            * in addition, extend test to evaluate model performance on all single classes from the dataset
        '''
        if self.test_method == 'batch':
            self.test_batch_mode(current_task, current_step, full_eval=self.full_eval)
            return

        else:
            if self.ml_paradigm == 'supervised' and self.single_class_test == 'yes':
                log.info('{:19s}'.format('\t SINGLE CLASS TEST').center(128, '-'))
                for test_task in range(0, self.single_cls_test.__len__()):
                    self.model.test_task = f'M{test_task}'  # test task identifier
                    test_dataset, test_steps = self.single_cls_test[test_task]  # get dataset & num of test steps f/e single class
                    log.info(f'[TEST] -> {test_task}')
                    self.model.evaluate(x=test_dataset[0], y=test_dataset[1],
                                        batch_size=self.test_batch_size,
                                        steps=test_steps,
                                        callbacks=self.eval_callbacks,
                                        verbose=1,
                                        return_dict=True)
        super()._test(current_task, full_eval=self.full_eval)


    def test_batch_mode(self, current_task, current_step, full_eval=False):
        #-------------------------- SINGLE CLASS TEST
        if self.ml_paradigm == 'supervised' and self.single_class_test == 'yes':
            log.info('{:19s}'.format('SINGLE CLASS TEST').center(128, '-'))
            for test_task in range(0, self.single_cls_test.__len__()):
                task_name                   = f'M{test_task}'  # test task identifier
                self.model.test_task        = task_name
                test_dataset, test_steps    = self.single_cls_test[test_task]  # get dataset & num of test steps f/e single class
                xs, ys                      = test_dataset[0], test_dataset[1]

                #-------------------------- START
                for e_cb in self.eval_callbacks: e_cb.on_test_begin()

                meaned_metrics = {}
                for t_s in range(0, test_steps):
                    lo          = t_s*self.test_batch_size
                    up          = t_s*self.test_batch_size+self.test_batch_size 
                    test_batch  = xs[lo:up], ys[lo:up]

                    for e_cb in self.eval_callbacks: e_cb.on_test_batch_begin(t_s)

                    metric_results = self.model.test_step(data=test_batch, return_dict=True)

                    for m_k, m_v in metric_results.items():
                        if m_k in meaned_metrics:
                            meaned_metrics[m_k] += m_v
                        else:
                            meaned_metrics.update({m_k : m_v})

                    for e_cb in self.eval_callbacks: e_cb.on_test_batch_end(t_s)

                metric_str = ''
                for m_k, m_v in meaned_metrics.items(): # log meaned metrics over all mini-batches
                    meaned_metrics[m_k] = m_v / test_steps
                    metric_str += '{:16s} {:8.2f}'.format(m_k, meaned_metrics[m_k]) + '\t'
                log.info(f'[TEST: CLASS] -> TASK: {test_task}')
                log.info(f'\tSTEP\t{t_s}:\t' + f'{metric_str}')

                if self.wandb_active == 'yes':
                    wandb_metrics_group = f't{current_task}_eval_{task_name}'
                    wandb.log({ wandb_metrics_group : meaned_metrics }, step=current_step)

                for e_cb in self.eval_callbacks: e_cb.on_test_end()
                #-------------------------- END
                self.model.reset_metrics()

        #-------------------------- ALL TASKS TEST
        if full_eval == True: task_end = self.num_tasks + 1
        else: task_end = current_task + 1

        log.info('{:16s}'.format(' ALL TASKS TEST').center(128, '-'))
        for test_task in range(0 if self.DAll != [] else 1, task_end):  # test: DAll, all previous, the current & (opt.) all future sub-tasks -> 0: DAll, 1: T1, 2: T2, ...
            task_name               = 'DAll' if test_task == 0 else f'T{test_task}'                     # build task name
            self.model.test_task    = task_name                                                         # test task identifier
            task_classes            = '(test classes: {})'.format(                                      # get tested classes
                ','.join(map(str, self.DAll if test_task == 0 else self.tasks[test_task])))

            test_dataset    = self.test_D_ALL if test_task == 0 else self.test_sets[test_task]          # get dataset to test
            test_steps      = (self.samples_test_D_ALL // self.test_batch_size) if test_task == 0 else self.get_task_iterations(test_task)[1]
            
            log.info(f'[TEST] -> {task_name} {task_classes}\t')
            log.debug(f'X: {test_dataset[0].shape} Y: {test_dataset[1].shape} CLS: {test_dataset[1].sum(axis=0)}\n')
            xs, ys          = test_dataset

            #-------------------------- START
            for e_cb in self.eval_callbacks: e_cb.on_test_begin()
            meaned_metrics = {}
            for t_s in range(0, test_steps):
                lo          = t_s*self.test_batch_size
                up          = t_s*self.test_batch_size+self.test_batch_size 
                test_batch  = xs[lo:up], ys[lo:up]

                for e_cb in self.eval_callbacks: e_cb.on_test_batch_begin(t_s)

                metric_results = self.model.test_step(data=test_batch, return_dict=True)
                for m_k, m_v in metric_results.items():
                    if m_k in meaned_metrics:
                        meaned_metrics[m_k] += m_v
                    else:
                        meaned_metrics.update({m_k : m_v})

                for e_cb in self.eval_callbacks: e_cb.on_test_batch_end(t_s)

            metric_str = ''
            for m_k, m_v in meaned_metrics.items():
                meaned_metrics[m_k] = m_v / test_steps
                metric_str += '{:16s} {:8.2f}'.format(m_k, meaned_metrics[m_k]) + '\t'
            log.info(f'\tSTEP\t{t_s}:\t' + f'{metric_str}')

            if self.wandb_active == 'yes':
                wandb_metrics_group = f't{current_task}_eval_{task_name}'
                wandb.log({ wandb_metrics_group : meaned_metrics }, step=current_step)

            for e_cb in self.eval_callbacks: e_cb.on_test_end()
            #-------------------------- END
            self.model.reset_metrics()


    def load_single_class_test(self):
        ''' Load the test data for all single classes when the experiment is loaded. '''
        self.single_cls_test = [tuple()] * len(self.DAll)
        for task in range(0, len(self.DAll)):
            _, test, _, samples_test    = self.dataset.get_dataset(classes=[task])
            test_iters                  = samples_test // self.test_batch_size
            self.single_cls_test[task]  = (test, test_iters)
