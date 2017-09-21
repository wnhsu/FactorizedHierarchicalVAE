from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops

def dense_latent(inputs,
                 num_outputs,
                 mu_nl=None,
                 logvar_nl=None,
                 normalizer_fn=None,
                 normalizer_params=None,
                 weights_initializer=initializers.xavier_initializer(),
                 weights_regularizer=None,
                 biases_initializer=init_ops.zeros_initializer(),
                 biases_regularizer=None,
                 reuse=None,
                 variables_collections=None,
                 outputs_collections=None,
                 trainable=True,
                 scope=None):
    """a latent variable layer"""
    # normalizer is disabled for now
    assert(normalizer_fn is None and normalizer_params is None)

    with tf.variable_scope(scope, "dense_latent", [inputs], reuse=reuse) as sc:
        mu = fully_connected(inputs,
                             num_outputs,
                             activation_fn=mu_nl,
                             normalizer_fn=normalizer_fn,
                             normalizer_params=normalizer_params,
                             weights_initializer=weights_initializer,
                             weights_regularizer=weights_regularizer,
                             biases_initializer=biases_initializer,
                             biases_regularizer=biases_regularizer,
                             reuse=reuse,
                             variables_collections=variables_collections,
                             outputs_collections=outputs_collections,
                             trainable=trainable,
                             scope="mu")
        logvar = fully_connected(inputs,
                                 num_outputs,
                                 activation_fn=logvar_nl,
                                 normalizer_fn=normalizer_fn,
                                 normalizer_params=normalizer_params,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer,
                                 biases_initializer=biases_initializer,
                                 biases_regularizer=biases_regularizer,
                                 reuse=reuse,
                                 variables_collections=variables_collections,
                                 outputs_collections=outputs_collections,
                                 trainable=trainable,
                                 scope="logvar")
        epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')
        sample = mu + tf.exp(0.5 * logvar) * epsilon

    return mu, logvar, sample

def deconv_latent(inputs,
                  num_outputs,
                  kernel_size,
                  stride,
                  padding,
                  data_format,
                  mu_nl=None,
                  logvar_nl=None,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  post_trim=None,
                  scope=None):
    """a deconvolutional latent variable layer"""
    # normalizer is disabled for now
    assert(normalizer_fn is None and normalizer_params is None)

    with tf.variable_scope(scope, "deconv_latent", [inputs], reuse=reuse) as sc:
        mu = conv2d_transpose(inputs,
                              num_outputs=num_outputs,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              data_format=data_format,
                              activation_fn=mu_nl,
                              normalizer_fn=normalizer_fn,
                              normalizer_params=normalizer_params,
                              weights_initializer=weights_initializer,
                              weights_regularizer=weights_regularizer,
                              biases_initializer=biases_initializer,
                              biases_regularizer=biases_regularizer,
                              reuse=reuse,
                              variables_collections=variables_collections,
                              outputs_collections=outputs_collections,
                              trainable=trainable,
                              scope="mu")
        logvar = conv2d_transpose(inputs,
                                  num_outputs=num_outputs,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  data_format=data_format,
                                  activation_fn=logvar_nl,
                                  normalizer_fn=normalizer_fn,
                                  normalizer_params=normalizer_params,
                                  weights_initializer=weights_initializer,
                                  weights_regularizer=weights_regularizer,
                                  biases_initializer=biases_initializer,
                                  biases_regularizer=biases_regularizer,
                                  reuse=reuse,
                                  variables_collections=variables_collections,
                                  outputs_collections=outputs_collections,
                                  trainable=trainable,
                                  scope="logvar")
        if post_trim:
            # print("before cropping: %s" % (mu.shape.as_list()))
            if data_format == "NCHW":
                mu = mu[..., post_trim[0], post_trim[1]]
                logvar = logvar[...,  post_trim[0], post_trim[1]]
            elif data_format == "NHWC":
                mu = mu[..., post_trim[0], post_trim[1], :]
                logvar = logvar[...,  post_trim[0], post_trim[1], :]
            else:
                raise ValueError("data_format %s not supported" % data_format)
        epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')
        sample = mu + tf.exp(0.5 * logvar) * epsilon

    return mu, logvar, sample
