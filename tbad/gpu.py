import os

import tensorflow as tf
import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def configure_gpu_resources(gpu_ids, gpu_memory_fraction):
    """Configure GPU resources for tensorflow.

    Configures which GPUs to use by tensorflow and the percentage of memory to grab from each GPU.

    Argument(s):
        gpu_ids -- A comma-separated string containing the GPU ids. For instance, '2,3' indicates that GPUs 2 and 3
            are available for use. GPU counting starts from 0.
        gpu_memory_fraction -- A float value between 0 and 1 specifying the amount of memory to grab from each GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
