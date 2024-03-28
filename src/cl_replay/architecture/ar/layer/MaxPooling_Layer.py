import tensorflow   as tf
import numpy        as np
import math

from itertools              import product

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log



class MaxPooling_Layer(Custom_Layer):
    ''' 
    Standard CNN max pooling, but together with backwards sampling.
        - If kernel size is not a divisor of input tensor h/w, input is zero-padded
        - This is not implemented physically but by manipulating the lookup structures
    '''
    def __init__(self, **kwargs):
        super(MaxPooling_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs
        self.input_layer            = self.parser.add_argument('--input_layer',             type=int,   default=[None], help=f'prefix integer(s) of this layer inputs')

        self.sharpening_rate        = self.parser.add_argument('--sharpening_rate',         type=float, default=.1,     help='if sampling is active, use sharpening rate to improve samples with gradient')
        self.sharpening_iterations  = self.parser.add_argument('--sharpening_iterations',   type=int,   default=100,    help='number of sharpening iterations')
        self.target_layer           = self.parser.add_argument('--target_layer',            type=int,   default=-1,     help='target GMM layer index for sharpening')

        self.kernel_size_y          = self.parser.add_argument('--kernel_size_y',           type=int,                   help='kernel width y')
        self.kernel_size_x          = self.parser.add_argument('--kernel_size_y',           type=int,                   help='kernel height x')
        self.kernel_size_t          = self.parser.add_argument('--kernel_size_t',           type=int,   default=1,      help='kernel size in temporal dimension, assuming that temporal dim is folde"d into the channel dimension')
        self.stride_y               = self.parser.add_argument('--stride_y',                type=int,   default=1,      help='stride y')
        self.stride_x               = self.parser.add_argument('--stride_x',                type=int,   default=1,      help='stride x')

        self.sampling_mode          = self.parser.add_argument('--sampling_mode',           type=str,   default='dense',help='dense or sparse', choices=['dense', 'sparse'])
        self.batch_size             = self.parser.add_argument('--batch_size',              type=int,   default=100,    help='bs')

        self.trainable = False


    def build(self, input_shape):
        self.w_in       = input_shape[1]
        self.h_in       = input_shape[2]
        self.c_in       = input_shape[3]

        # Set kernel size defaults if not specified via parser
        if self.kernel_size_y is None: self.kernel_size_y = self.h_in
        if self.kernel_size_x is None: self.kernel_size_x = self.w_in

        self.kernel_size_x  = self.kernel_size_x * self.kernel_size_t
        self.w_in           = self.w_in * self.kernel_size_t
        self.c_in           = self.c_in // self.kernel_size_t
        self.stride_x       = self.stride_x * self.kernel_size_t

        # compute output size, pad input if required!
        self.h_out = 1 + math.ceil((self.h_in - self.kernel_size_y) / self.stride_y)
        self.w_out = 1 + math.ceil((self.w_in - self.kernel_size_x) / self.stride_x)
        self.c_out = self.c_in

        log.debug(f'{self.name} input shape={input_shape}, kernel_y={self.kernel_size_y}, kernel_x={self.kernel_size_x}, stride_y={self.stride_y}, stride_x={self.stride_x}')
        log.debug(f'{self.name} h_out={self.h_out}, w_out={self.w_out}, c_out={self.c_out}')

        # pre-compute constants for pooling
        # -- for collecting input tensor values via tf.gather such that max can be taken
        lookupShape = [self.h_out, self.w_out, self.c_out, self.kernel_size_y, self.kernel_size_x]
        self.np_lookupArray         = np.zeros(lookupShape, dtype=np.int64)
        # -- remember indices of values to null before max taking since they come from outside the input and contain corrupted values. But should contain zeroes
        self.np_zeroMask            = np.ones(lookupShape, dtype=self.dtype_np_float)
        # -- for (up-)sampling
        self.np_invArray            = np.zeros([self.h_in, self.w_in, self.c_in])

        # construct constants
        # -- forward lookup
        # ---- loop over grid positions of filter w_indows in input
        for h, w, c in product(range(self.h_out), range(self.w_out), range(self.c_out)):
            # ---- loop over input pixels IN filter w_indows at a certain position
            for inPatchY, inPatchX in product(range(self.kernel_size_y), range(self.kernel_size_x)):
                inPatchStartY       = h * self.stride_y
                inPatchStartX       = w * self.stride_x
                inC                 = c % self.c_in
                inY                 = inPatchStartY + inPatchY
                inX                 = inPatchStartX + inPatchX

                if inY >= self.h_in or inX >= self.w_in:
                    self.np_lookupArray[h, w, c, inPatchY, inPatchX]    = 0
                    self.np_zeroMask[h, w, c, inPatchY, inPatchX]       = 0
                else:
                    self.np_lookupArray[h, w, c, inPatchY, inPatchX]    = self.w_in * self.c_in * inY + self.c_in * inX + inC
        self.lookupArray                = tf.constant(self.np_lookupArray.reshape((self.h_out * self.w_out * self.c_out * self.kernel_size_y * self.kernel_size_x)))
        self.zeroMaskArray              = tf.constant(self.np_zeroMask.reshape(1, -1))

        self.np_inv_arr                 = np.zeros([self.h_in * self.w_in * self.c_in], dtype=np.int64)                 # sampling lookup

        for inIndex, (h, w, c) in enumerate(product(range(self.h_in), range(self.w_in), range(self.c_in))):
            outY                        = h // self.kernel_size_y
            outX                        = w // self.kernel_size_x
            outC                        = c
            outIndex                    = outY * self.w_out * self.c_out + outX * self.c_out + outC
            self.np_inv_arr[inIndex]    = outIndex

        self.invArr                     = tf.constant(self.np_inv_arr)

        shufflingMask       = np.ones([self.h_out * self.w_out, self.kernel_size_x * self.kernel_size_y]) * -1.
        patchSize           = self.kernel_size_x * self.kernel_size_y

        #TODO: correct for fact that border patches may have their ones outside the input tensor
        # two ways of sampling throughtmaxcpooling:
        # a) repeat topdown value to all pixels in patch
        # b) repat topdown value to single, random position in patch, otherwise 0.0
        # create a tensor that serves as mask for the generated input in case of a)
        for c in range(0, self.h_out * self.w_out):
            offset                      = c % patchSize
            shufflingMask[c, offset]    = 1.0

        self.shuffling_mask = tf.constant(shufflingMask, dtype=self.dtype_tf_float)


    def call(self, inputs, training=None, *args, **kwargs):
        input_data      = inputs
        self.fwd        = self.forward(input_tensor=input_data)

        return self.fwd


    #@tf.function(autograph=False)
    def forward(self, input_tensor):
        # try a simple approach analogous to folding layer: use gather to copy all 2x2 patches to a continuous channel dimension
        folded_tensor    = tf.reshape(input_tensor, (self.batch_size, -1))
        folded_tensor    = tf.gather(folded_tensor, self.lookupArray, axis=1)
        # foldedTensorMasked = foldedTensor * self.zeroMaskArray
        folded_tensor    = tf.reshape(folded_tensor,(-1, self.h_out, self.w_out, self.c_out, self.kernel_size_y * self.kernel_size_x))
        max_op           = tf.reduce_max(folded_tensor, axis=4)

        return max_op


    def get_fwd_result(self): return self.fwd


    def backwards(self, topdown, *args, **kwargs):
        tmp = tf.reshape(
            tf.gather(tf.reshape(topdown, (-1, self.h_out * self.w_out * self.c_out)), self.invArr, axis=1),
            (-1, self.h_in, self.w_in // self.kernel_size_t, self.c_in * self.kernel_size_t)
        )
        log.debug(f" h_in, w_in, c_in={self.h_in}, {self.w_in}, {self.c_in}, topdown shape={topdown.shape}, to lower shape={tmp.shape}")

        # TODO: does this work with sp.temporal pooling (kernel_size_t > 1)
        if self.sampling_mode == "sparse":
            mask1 = tf.random.shuffle(self.shuffling_mask)                                                              # h_out * w_out, ksX * ksY
            mask2 = tf.reshape(mask1, [1, self.h_out, self.w_out, self.kernel_size_y, self.kernel_size_x])              # -> 1, h_out, w_out, ksY, ksX
            mask3 = tf.transpose(mask2, [0, 1, 3, 2, 4])                                                                # -> 1, h_out, ksY, w_out, ksX
            mask4 = tf.reshape(mask3, [1, self.h_out * self.kernel_size_y, self.w_out * self.kernel_size_x,1])          # -> 1, h_out*ksY, w_out*ksX

            return tmp * mask4[:, 0:self.h_in, 0:self.w_in, :]
        else:
            return tmp


    def get_target_layer(self):             return self.target_layer
    def get_sharpening_iterations(self):    return self.sharpening_iterations
    def get_sharpening_rate(self):          return self.sharpening_rate


    def compute_output_shape(self, input_shape):
        ''' returns a tuple containing the output shape of this layers computation '''
        return self.batch_size, self.h_out, self.w_out, self.c_out
