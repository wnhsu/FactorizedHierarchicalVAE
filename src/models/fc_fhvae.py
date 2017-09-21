"""Fully-Connected Factorized Hierarchical VAE Class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import nn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer

from libs.layers import dense_latent, cat_dense_latent
from models import *
from models.base_fhvae import BaseFacHierVAE

class FCFacHierVAE(BaseFacHierVAE):
    def __init__(self, model_conf, train_conf, **kwargs):
        info("FCFacHierVAE constructor:\n%s" % (locals()))
        info("unused kwargs: %s" % kwargs)
        super(FCFacHierVAE, self).__init__(model_conf, train_conf)

    def _set_model_conf(self, model_conf):
        """
        Args:
            - model_conf: specifies the model configurations.
                - input_shape: list of (c, h, w)
                - input_dtype: input data type
                - target_shape: list of (c, h, w)
                - target_dtype: target data type
                - hu_z1_enc: list of int of number of hidden 
                    units at each layer for z1_encoder
                - hu_z2_enc: list of int of number of hidden 
                    units at each layer for z2_encoder
                - hu_dec: list of int of number of hidden 
                    units at each layer for decoder. do not
                    specify if using symmetric architecture
                - n_latent1: number of latent variables 
                    with partial supervision
                - n_latent2: number of latent variables 
                    without supervision
                - n_class1: number of different mu1 in the 
                    supervised partition
                - latent1_std: std for p(z1 | mu1)
                - z1_logvar_nl:
                - z2_logvar_nl: 
                - x_conti: whether target is continuous or discrete
                    use Gaussian for continuous target, softmax
                    for discrete target
                - x_mu_nl: activation function for target mean
                - x_logvar_nl: activation function for target 
                    log variance
                - n_bins: discretized target dimension
                - if_bn: use batch normalization if True
        """
        self._model_conf = {"input_shape": None,
                            "input_dtype": tf.float32,
                            "target_shape": None,
                            "target_dtype": tf.float32,
                            "hu_z1_enc": [],
                            "hu_z2_enc": [],
                            "hu_dec": [],
                            "n_latent1": 64,
                            "n_latent2": 64,
                            "n_class1": None,
                            "latent1_std": 0.5,
                            "z1_logvar_nl": None,
                            "z2_logvar_nl": None,
                            "x_conti": True,
                            "x_mu_nl": None,
                            "x_logvar_nl": None,
                            "n_bins": None,
                            "if_bn": True}
        for k in model_conf:
            if k in self._model_conf:
                self._model_conf[k] = model_conf[k]
            else:
                raise ValueError("invalid model_conf: %s" % str(k))
        
        print("=" * 20)
        print("MODEL CONF:")
        for k, v in self._model_conf.items():
            print("%s: %s" % (k, v))
        print("=" * 20)
            
    def _build_z1_encoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._train_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"scope": "BatchNorm",
                                 "is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}
            # TODO: need to upgrade to latest, 
            #       which commit support param_regularizers args

        input_dim = np.prod(inputs.get_shape().as_list()[1:])
        outputs = tf.reshape(inputs, [-1, input_dim])
        with tf.variable_scope("z1_enc", reuse=reuse):
            for i, hu in enumerate(self._model_conf["hu_z1_enc"]):
                outputs = fully_connected(inputs=outputs,
                                          num_outputs=hu,
                                          activation_fn=nn.relu,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          weights_regularizer=weights_regularizer,
                                          reuse=reuse,
                                          scope="z1_enc_fc%s" % i)

            z1_mu, z1_logvar, z1 = dense_latent(
                    outputs, self._model_conf["n_latent1"], 
                    logvar_nl=self._model_conf["z1_logvar_nl"],
                    reuse=reuse, scope="z1_enc_lat")

        return [z1_mu, z1_logvar], z1

    def _build_z2_encoder(self, inputs, z1, reuse=False):
        weights_regularizer = l2_regularizer(self._train_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"scope": "BatchNorm",
                                 "is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}
            # TODO: need to upgrade to latest, 
            #       which commit support param_regularizers args

        input_dim = np.prod(inputs.get_shape().as_list()[1:])
        outputs = tf.concat([tf.reshape(inputs, [-1, input_dim]), z1], axis=1)
        with tf.variable_scope("z2_enc", reuse=reuse):
            for i, hu in enumerate(self._model_conf["hu_z2_enc"]):
                outputs = fully_connected(inputs=outputs,
                                          num_outputs=hu,
                                          activation_fn=nn.relu,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          weights_regularizer=weights_regularizer,
                                          reuse=reuse,
                                          scope="z2_enc_fc%s" % i)

            z2_mu, z2_logvar, z2 = dense_latent(
                    outputs, self._model_conf["n_latent2"],
                    logvar_nl=self._model_conf["z2_logvar_nl"],
                    reuse=reuse, scope="z2_enc_lat")
        
        return [z2_mu, z2_logvar], z2

    def _build_decoder(self, z1, z2, reuse=False):
        weights_regularizer = l2_regularizer(self._train_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"scope": "BatchNorm",
                                 "is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}
            # TODO: need to upgrade to latest, which 
            #       commit support param_regularizers args

        outputs = tf.concat([z1, z2], axis=1)
        with tf.variable_scope("dec", reuse=reuse):
            for i, hu in enumerate(self._model_conf["hu_dec"]):
                outputs = fully_connected(inputs=outputs,
                                          num_outputs=hu,
                                          activation_fn=nn.relu,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          weights_regularizer=weights_regularizer,
                                          reuse=reuse,
                                          scope="dec_fc%s" % i)
            
            target_shape = list(self._model_conf["target_shape"])
            target_dim = np.prod(target_shape)

            if self._model_conf["x_conti"]:
                mu_nl = self._model_conf["x_mu_nl"]
                logvar_nl = self._model_conf["x_logvar_nl"]
                x_mu, x_logvar, x = dense_latent(outputs,
                                                 target_dim,
                                                 mu_nl=mu_nl,
                                                 logvar_nl=logvar_nl,
                                                 reuse=reuse,
                                                 scope="dec_lat")
                x_mu = tf.reshape(x_mu, [-1]+target_shape)
                x_logvar = tf.reshape(x_logvar, [-1] + target_shape)
                px = [x_mu, x_logvar]
            else:
                n_bins = self._model_conf["n_bins"]
                x_logits, x = cat_dense_latent(
                        outputs, target_dim, n_bins, reuse=reuse, scope="dec_lat")
                x_logits = tf.reshape(x_logits, [-1] + target_shape + [n_bins])
                px = x_logits

            x = tf.reshape(x, [-1] + target_shape)

        return px, x
