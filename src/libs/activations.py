from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import functools

def custom_relu(x, cutoff=0):
    if cutoff == 0:
        return 0.5 * (x + tf.abs(x))
    else:
        return 0.5 * ((x - cutoff) + tf.abs(x - cutoff)) + cutoff

def custom_double_relu(x, scale):
    assert(scale > 0)
    return custom_relu(x, -1 * scale) - custom_relu(x, scale) + scale

relu_n10 = functools.partial(custom_relu, cutoff=-10)
