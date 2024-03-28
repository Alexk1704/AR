import os
import sys
import logging
import warnings

from absl import logging as absl_logging

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

fmt = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s'
formatter = logging.Formatter(fmt)

absl_logging.get_absl_handler().setFormatter(formatter)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tf debug messages

import tensorflow as tf

#--------------------------------------------------------------------------------

print("TF version:",            tf.version.VERSION)
print("TF executing eagerly:",  tf.executing_eagerly())
print("TF built with CUDA:",    tf.test.is_built_with_gpu_support())

gpus = tf.config.experimental.list_physical_devices('GPU')
if not gpus:
    print('No valid gpu found', file=sys.stderr)
else:
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        gpu_name = details.get('device_name', 'Unknown GPU')
        comp_cap = details.get('compute_capability', 'Unknown compute capability')
        print(f'{gpu}: {gpu_name}, Compute Capability: {comp_cap}')
        try:    tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

#--------------------------------------------------------------------------------

for h in tf.get_logger().handlers:
    h.setFormatter(formatter)

tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(10)

#--------------------------------------------------------------------------------

""" 
DebuggerV2
    * see https://www.tensorflow.org/tensorboard/debugger_v2 
    * live logging via: $ python3 -m tensorboard.main --logdir=/tmp/tfdbg2_logdir

tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
"""
