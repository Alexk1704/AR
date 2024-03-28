from cl_replay.api.parsing   import Kwarg_Parser


class Supervised_Replay_Adaptor: 
    '''
        Supervised_Replay_Adaptor: Externalize data generation for a supervised scenario.

        Attributes
        ----------
        samples_to_generate : float, default=1000.
            - The number of samples for data generation.
        sampling_batch_size : int, default=100
            - Batch size for sampling
        replay_strategy : str, ['buffer', 'generator']
            - Set the replay strategy, either relying on a buffer or generator
        replay_proportions : list, default=[50.,50.]
            - Define replay proportions for generated/real samples.
        
        ...
    '''
    def __init__(self, **kwargs):
        self.parser = Kwarg_Parser(**kwargs)
        self.samples_to_generate    = self.parser.add_argument('--samples_to_generate', type=float, default=1000.,      help='num of samples for data generation')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,        help='batch size for sampling')
        self.replay_strategy        = self.parser.add_argument('--replay_strategy',     type=str,   default='generator', choices=['buffer', 'generator'], help='set the replay strategy, either relying on a buffer or generator')
        self.replay_proportions     = self.parser.add_argument('--replay_proportions',  type=float, default=[50., 50.], help='define replay proportions for generated/real samples')


    def set_input_dims(self, h, w, c, num_classes):
        ''' Set input dimension to be used for data generation. '''
        self.h = h
        self.w = w
        self.c = c
        self.num_classes = num_classes


    def get_input_dims(self):
        ''' Get input dimension to be used for data generation. '''
        return self.h, self.w, self.c, self.num_classes


    def set_model(self, model):
        ''' Set a model instance for this adaptor object. '''
        self.model = model


    def set_buffer(self, buffer):
        ''' Set a buffer instance for this adaptor object. '''
        self.buffer = buffer

    
    def sample_from_buffer(self, p_id, sample_size, **kwargs): #TODO: replay buffer
        ''' Samples from a given partition. '''
        pass


    def set_generator(self, generator):
        ''' Set a generator instance for this adaptor object. '''
        self.generator = generator


    def generate(self, task=-1, classes=None, xs=None, **kwargs):
        '''
            Starts a (model/architecture-specific) data generation procedure.

            Parameters
            ----------
            task : int, optional, default=-1
                - Task identifier.
            classes : list, optional
                - A list of classes for conditional sampling.
            xs : np.array, tf.constant, optional, default=None
                - The current input data.

            Returns
            -------
            - (gen_samples, gen_labels) : tuple
                - Returns the generated samples as a tuple of numPy arrays.

            ...
        '''
        pass


    def before_subtask(self, task=-1, **kwargs):
        ''' Performs pre sub-task actions. '''
        pass


    def after_subtask(self, task=-1, **kwargs):
        ''' Perform past sub-task actions. '''
        pass