from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def _get_conv_output_shape(inp_shape, conv):
    """compute output shape of one convolutional layer
    
    inp_shape: (..., c, h, w)
    conv: (c, kernel_h, kernel_w, stride_h, stride_w, padding)
    """
    h, w = inp_shape[-2:]
    conv_c, kernel_h, kernel_w, stride_h, stride_w, padding = conv 
    assert(padding in {"same", "valid"})
    assert(h >= kernel_h and w >= kernel_w)
    if padding == "same":
        output_h = int(np.ceil(float(h) / float(stride_h)))
        output_w = int(np.ceil(float(w) / float(stride_w)))
    else:
        output_h = int(np.ceil(float(h - kernel_h + 1) / float(stride_h)))
        output_w = int(np.ceil(float(w - kernel_w + 1) / float(stride_w)))
    return tuple(inp_shape[:-3]) + (conv_c, output_h, output_w)


def get_conv_output_shape(inp_shape, convs):
    """compute output shape after a list of convolutional layers"""
    output_shape = inp_shape
    for conv in convs:
        output_shape = _get_conv_output_shape(output_shape, conv)
    return output_shape
