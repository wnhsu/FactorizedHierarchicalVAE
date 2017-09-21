from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def kld(mu, logvar, q_mu=None, q_logvar=None):
    """compute dimension-wise KL-divergence
    -0.5 (1 + logvar - q_logvar - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar))
    q_mu, q_logvar assumed 0 is set to None
    """
    if q_mu is None:
        q_mu = tf.zeros_like(mu)
    else:
        print("using non-default q_mu %s" % q_mu)

    if q_logvar is None:
        q_logvar = tf.zeros_like(logvar)
    else:
        print("using non-default q_logvar %s" % q_logvar)

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(q_mu, tf.Tensor):
        q_mu_shape = q_mu.get_shape().as_list()
    else:
        q_mu_shape = list(np.asarray(q_mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)

    if isinstance(q_logvar, tf.Tensor):
        q_logvar_shape = q_logvar.get_shape().as_list()
    else:
        q_logvar_shape = list(np.asarray(q_logvar).shape)

    if not np.all(mu_shape == logvar_shape):
        raise ValueError("mu_shape (%s) and logvar_shape (%s) does not match" % (
            mu_shape, logvar_shape))
    if not np.all(mu_shape == q_mu_shape):
        raise ValueError("mu_shape (%s) and q_mu_shape (%s) does not match" % (
            mu_shape, q_mu_shape))
    if not np.all(mu_shape == q_logvar_shape):
        raise ValueError("mu_shape (%s) and q_logvar_shape (%s) does not match" % (
            mu_shape, q_logvar_shape))
    
    return -0.5 * (1 + logvar - q_logvar - \
            (tf.pow(mu - q_mu, 2) + tf.exp(logvar)) / tf.exp(q_logvar))

def log_gauss(mu, logvar, x):
    """compute point-wise log prob of Gaussian"""
    x_shape = x.get_shape().as_list()

    if isinstance(mu, tf.Tensor):
        mu_shape = mu.get_shape().as_list()
    else:
        mu_shape = list(np.asarray(mu).shape)

    if isinstance(logvar, tf.Tensor):
        logvar_shape = logvar.get_shape().as_list()
    else:
        logvar_shape = list(np.asarray(logvar).shape)
    
    if not np.all(x_shape == mu_shape):
        raise ValueError("x_shape (%s) and mu_shape (%s) does not match" % (
            x_shape, mu_shape))
    if not np.all(x_shape == logvar_shape):
        raise ValueError("x_shape (%s) and logvar_shape (%s) does not match" % (
            x_shape, logvar_shape))

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.pow((x - mu), 2) / tf.exp(logvar))

def log_normal(x):
    """compute point-wise log prob of Gaussian"""
    return -0.5 * (np.log(2 * np.pi) + tf.pow(x, 2))
