import tensorflow   as tf
import numpy        as np

from itertools      import product

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log



class Folding_Layer(Custom_Layer):


    def __init__(self, **kwargs):
        super(Folding_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs
        self.input_layer            = self.parser.add_argument('--input_layer',             type=int,   default=[None], help=f'prefix integer(s) of this layer inputs')
        self.batch_size             = self.parser.add_argument('--batch_size',              type=int,   default=100,    help='bs')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size',     type=int,   default=100,    help='sampling batch size')

        self.patch_height           = self.parser.add_argument('--patch_height',            type=int,   default=-1,     help='patch height')
        self.patch_width            = self.parser.add_argument('--patch_width',             type=int,   default=-1,     help='patch width')
        self.stride_y               = self.parser.add_argument('--stride_y',                type=int,   default=-1,     help='stride y')
        self.stride_x               = self.parser.add_argument('--stride_x',                type=int,   default=-1,     help='stride x')

        self.trainable = False


    def build(self, input_shape):
        self.h_in       = input_shape[1]
        self.w_in       = input_shape[2]
        self.c_in       = input_shape[3]

        if not isinstance(self.h_in, int): # compat
            self.h_in = self.h_in.value
            self.w_in = self.w_in.value
            self.c_in = self.c_in.value

        if self.patch_height == -1  : self.patch_height     = self.h_in
        if self.patch_width == -1   : self.patch_width      = self.w_in
        if self.stride_y == -1      : self.stride_y         = self.h_in
        if self.stride_x == -1      : self.stride_x         = self.w_in

        self.h_out          = int((self.h_in - self.patch_height) / (self.stride_y) + 1)
        self.w_out          = int((self.w_in - self.patch_width) / (self.stride_x) + 1)
        self.c_out          = self.patch_height * self.patch_width * self.c_in
        self.output_size    = self.h_out * self.w_out * self.c_out

        self.indicesOneSample   = np.zeros([int(self.h_out * self.w_out * self.c_out)], dtype=np.int32)  # for forward transmission
        ''' for backward transmission (sampling) '''
        mapCorr                 = np.zeros([1, self.h_in, self.w_in, 1])
        indexArr                = np.zeros([self.sampling_batch_size, self.h_out * self.w_out * self.c_out], dtype=np.int32)
        '''
        Create forward & backward indices arrays for use with gather (forward) and scatter (backward) loop over all filter positions in output layer. 
        Compute unique index of corresponding pixel in input tensor, and fill arrays
        '''
        for outIndex, (outY, outX, outC) in enumerate(product(range(self.h_out), range(self.w_out), range(self.c_out))):
            inFilterY                       = outY * self.stride_y
            inFilterX                       = outX * self.stride_x
            inC                             = outC % self.c_in
            inCFlatIndex                    = outC // self.c_in
            inY                             = inFilterY + inCFlatIndex // self.patch_width
            inX                             = inFilterX + inCFlatIndex % self.patch_width
            inIndex                         = inY * self.w_in * self.c_in + inX * self.c_in + inC
            self.indicesOneSample[outIndex] = inIndex
            indexArr[:, outIndex]           = inIndex
            if inC == 0: mapCorr[0, inY, inX, 0] += 1
        indexArr        += np.array([[i * self.h_in * self.w_in * self.c_in] for i in range(self.sampling_batch_size)])
        self.indexArr   = tf.constant(indexArr, dtype=tf.int32)
        self.mapCorr    = tf.constant(mapCorr, dtype=tf.float32)
        acc_shape       = (self.sampling_batch_size * self.h_in * self.w_in * self.c_in)
        self.acc        = tf.Variable(initial_value=np.zeros(acc_shape), dtype=tf.float32, name='acc', trainable=False)


    def call(self, inputs, training=None, *args, **kwargs):
        train_data          = inputs
        self.fwd            = self.forward(input_tensor=train_data)

        return self.fwd


    def forward(self, input_tensor):
        ''' Transforms all samples at the same time by axis=1 arg to gather. '''
        gatherRes   = tf.gather(tf.reshape(input_tensor, (-1, self.h_in * self.w_in * self.c_in)), self.indicesOneSample, axis=1)
        convert_op  = tf.reshape(gatherRes, (-1, self.w_out, self.h_out, self.c_out))

        return convert_op


    def get_fwd_result(self): return self.fwd


    def backwards(self, topdown, *args, **kwargs):
        self.acc.assign(self.acc * 0.0)
        self.acc.scatter_add(
            tf.IndexedSlices(tf.reshape(topdown, (-1,)),
                             tf.cast(tf.reshape(self.indexArr, -1), self.dtype_tf_int))
        )
        backProj = tf.reshape(self.acc, (self.sampling_batch_size, self.h_in, self.w_in, self.c_in))
        sampling_op = backProj / self.mapCorr

        return sampling_op



    def compute_output_shape(self, input_shape):    return self.batch_size, self.h_out, self.w_out, self.c_out


    def get_logging_params(self):
        return { f'{self.prefix}patch_height'   : self.patch_height,
                 f'{self.prefix}patch_width'    : self.patch_width,
                 f'{self.prefix}stride_x'       : self.stride_x,
                 f'{self.prefix}stride_y'       : self.stride_y,
        }
