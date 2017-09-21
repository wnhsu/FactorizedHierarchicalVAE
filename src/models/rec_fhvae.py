"""Recurrent Factorized Hierarchical VAE Class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import \
        BasicRNNCell, GRUCell, BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn

from libs.layers import dense_latent
from models import *
from models.base_fhvae import BaseFacHierVAE

_cell_dict = {"rnn": BasicRNNCell,
              "gru": GRUCell,
              "lstm": BasicLSTMCell}

class RecFacHierVAE(BaseFacHierVAE):
    def __init__(self, model_conf, train_conf, training=False, **kwargs):
        info("=" * 20)
        info("RecFacHierVAE constructor:")
        info("training mode: %s" % training)
        info("unused kwargs: %s" % kwargs)
        info("=" * 20)
        self.training = training
        super(RecFacHierVAE, self).__init__(model_conf, train_conf)

    def _set_model_conf(self, model_conf):
        """
        Args:
            - model_conf: specifies the model configurations.
                - input_shape: list of (c, h, w)
                - input_dtype: input data type
                - target_shape: list of (c, h, w)
                - target_dtype: target data type
                - rec_cell_type: "rnn", "gru" or "lstm"
                - rec_learn_init: 
                - rec_z1_enc: list of number of recurrent hidden units
                    at each layer for encoder
                - rec_z1_enc_concur: #time steps input concurrently 
                    (for now total time steps must be multiples 
                    of `rec_dec_concur`)
                - rec_z1_enc_out: what from recurrent encoder to 
                    output to next module (fully-connected or latent).
                    accept "{last,all}_{h,c,hc}," where last
                    means only take from the last recurrent layer, 
                    and h, c refers to the output and cell state from
                    the last time step respectively
                - rec_z1_enc_bi: use bi-directional rnn for encoder if True
                - hu_z1_enc: list of int of number of hidden 
                    units at each layer for z1_encoder
                - rec_z2_enc: 
                - rec_z2_enc_concur: 
                - rec_z2_enc_out: 
                - rec_z2_enc_bi: 
                - hu_z2_enc: 
                - hu_dec: list of int of number of hidden 
                    units at each layer for decoder. do not
                    specify if using symmetric architecture
                - rec_dec: list of number of recurrent hidden units
                    at each layer for decoder
                - rec_dec_bi: use bi-directional rnn for decoder if True
                - rec_dec_inp_train: feed target/prediction of the previous
                    time step to the recurrent decoder as input when 
                    predicting the distribution of the target of the next
                    time step. choices = {"", "targets", "x_mu", "x"}, ""
                    for not feeding anything
                - rec_dec_inp_test: same as rec_dec_inp_train. choices =
                    {"", "x_mu", "x"}. must be None if rec_dec_inp_train
                    is None; and the others otherwise
                - rec_dec_concur: #time steps output concurrently 
                    (for now total time steps must be multiples 
                    of `rec_dec_concur`)
                - rec_dec_inp_hist: #previous time steps to condition on
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
                            "rec_cell_type": "gru",
                            "rec_learn_init": False,
                            "rec_z1_enc": [],
                            "rec_z1_enc_concur": 1,
                            "rec_z1_enc_out": "last_hc",
                            "rec_z1_enc_bi": False,
                            "hu_z1_enc": [],
                            "rec_z2_enc": [],
                            "rec_z2_enc_concur": 1,
                            "rec_z2_enc_out": "last_hc",
                            "rec_z2_enc_bi": False,
                            "hu_z2_enc": [],
                            "hu_dec": [],
                            "rec_dec": [],
                            "rec_dec_bi": False,
                            "rec_dec_inp_train": None,
                            "rec_dec_inp_test": None,
                            "rec_dec_concur": 1,
                            "rec_dec_inp_hist": 1,
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
        
        info("=" * 20)
        info("MODEL CONF:")
        for k, v in self._model_conf.items():
            info("%s: %s" % (k, v))
        info("=" * 20)
            
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

        if not hasattr(self, "_debug_outputs"):
            self._debug_outputs = {}

        C, T, F = self._model_conf["target_shape"]
        n_concur = self._model_conf["rec_z1_enc_concur"]
        if T % n_concur != 0:
            raise ValueError("total time steps must be multiples of %s" % (
                n_concur))
        n_frame = T // n_concur
        info("z1_encoder: n_frame=%s, n_concur=%s" % (n_frame, n_concur))

        with tf.variable_scope("z1_enc", reuse=reuse):
            # recurrent layers
            if self._model_conf["rec_z1_enc"]:
                # reshape to (N, n_frame, n_concur*C*F)
                inputs = array_ops.transpose(inputs, (0, 2, 1, 3))
                inputs_shape = inputs.get_shape().as_list()
                inputs_depth = np.prod(inputs_shape[2:])
                new_shape = (-1, n_frame,  n_concur * inputs_depth)
                inputs = tf.reshape(inputs, new_shape)

                self._debug_outputs["inp_reshape"] = inputs
                if self._model_conf["rec_z1_enc_bi"]:
                    raise NotImplementedError
                else:
                    Cell = _cell_dict[self._model_conf["rec_cell_type"]]
                    cell = MultiRNNCell([Cell(hu) \
                            for hu in self._model_conf["rec_z1_enc"]])

                    if self._model_conf["rec_learn_init"]:
                        raise NotImplementedError
                    else:
                        input_shape = tuple(array_ops.shape(input_) \
                                for input_ in nest.flatten(inputs))
                        batch_size = input_shape[0][0]
                        init_state = cell.zero_state(
                                batch_size, self._model_conf["input_dtype"])

                    _, final_states = dynamic_rnn(
                            cell, 
                            inputs,
                            dtype=self._model_conf["input_dtype"],
                            initial_state=init_state,
                            time_major=False,
                            scope="z1_enc_%sL_rec" % len(
                                self._model_conf["rec_z1_enc"]))
                    self._debug_outputs["raw_rnn_out"] = _
                    self._debug_outputs["raw_rnn_final"] = final_states

                    if self._model_conf["rec_z1_enc_out"].startswith("last"):
                        final_states = final_states[-1:]

                    if self._model_conf["rec_cell_type"] == "lstm":
                        outputs = []
                        for state in final_states:
                            if "h" in self._model_conf["rec_z1_enc_out"].split("_")[1]:
                                outputs.append(state.h)
                            if "c" in self._model_conf["rec_z1_enc_out"].split("_")[1]:
                                outputs.append(state.c)
                    else:
                        outputs = final_states

                    outputs = tf.concat(outputs, axis=-1)
                    self._debug_outputs["concat_rnn_out"] = outputs
            else:
                outputs = inputs

            # fully connected layers
            output_dim = np.prod(outputs.get_shape().as_list()[1:])
            outputs = tf.reshape(outputs, [-1, output_dim])

            for i, hu in enumerate(self._model_conf["hu_z1_enc"]):
                outputs = fully_connected(inputs=outputs,
                                          num_outputs=hu,
                                          activation_fn=nn.relu,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          weights_regularizer=weights_regularizer,
                                          reuse=reuse,
                                          scope="z1_enc_fc%s" % (i + 1))

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

        if not hasattr(self, "_debug_outputs"):
            self._debug_outputs = {}

        C, T, F = self._model_conf["target_shape"]
        n_concur = self._model_conf["rec_z2_enc_concur"]
        if T % n_concur != 0:
            raise ValueError("total time steps must be multiples of %s" % (
                n_concur))
        n_frame = T // n_concur
        info("z2_encoder: n_frame=%s, n_concur=%s" % (n_frame, n_concur))

        # input_dim = np.prod(inputs.get_shape().as_list()[1:])
        # outputs = tf.concat([tf.reshape(inputs, [-1, input_dim]), z1], axis=1)

        with tf.variable_scope("z2_enc", reuse=reuse):
            # recurrent layers
            if self._model_conf["rec_z2_enc"]:
                # reshape to (N, n_frame, n_concur*C*F)
                inputs = array_ops.transpose(inputs, (0, 2, 1, 3))
                inputs_shape = inputs.get_shape().as_list()
                inputs_depth = np.prod(inputs_shape[2:])
                new_shape = (-1, n_frame,  n_concur * inputs_depth)
                inputs = tf.reshape(inputs, new_shape)

                # append z1 to each frame
                tiled_z1 = tf.tile(tf.expand_dims(z1, 1), (1, n_frame, 1))
                inputs = tf.concat([inputs, tiled_z1], axis=-1)

                self._debug_outputs["inp_reshape"] = inputs
                if self._model_conf["rec_z2_enc_bi"]:
                    raise NotImplementedError
                else:
                    Cell = _cell_dict[self._model_conf["rec_cell_type"]]
                    cell = MultiRNNCell([Cell(hu) \
                            for hu in self._model_conf["rec_z2_enc"]])

                    if self._model_conf["rec_learn_init"]:
                        raise NotImplementedError
                    else:
                        input_shape = tuple(array_ops.shape(input_) \
                                for input_ in nest.flatten(inputs))
                        batch_size = input_shape[0][0]
                        init_state = cell.zero_state(
                                batch_size, self._model_conf["input_dtype"])

                    _, final_states = dynamic_rnn(
                            cell, 
                            inputs,
                            dtype=self._model_conf["input_dtype"],
                            initial_state=init_state,
                            time_major=False,
                            scope="z2_enc_%sL_rec" % len(
                                self._model_conf["rec_z2_enc"]))
                    self._debug_outputs["raw_rnn_out"] = _
                    self._debug_outputs["raw_rnn_final"] = final_states

                    if self._model_conf["rec_z2_enc_out"].startswith("last"):
                        final_states = final_states[-1:]

                    if self._model_conf["rec_cell_type"] == "lstm":
                        outputs = []
                        for state in final_states:
                            if "h" in self._model_conf["rec_z2_enc_out"].split("_")[1]:
                                outputs.append(state.h)
                            if "c" in self._model_conf["rec_z2_enc_out"].split("_")[1]:
                                outputs.append(state.c)
                    else:
                        outputs = final_states

                    outputs = tf.concat(outputs, axis=-1)
                    self._debug_outputs["concat_rnn_out"] = outputs
            else:
                input_dim = np.prod(inputs.get_shape().as_list()[1:])
                outputs = tf.concat([tf.reshape(inputs, [-1, input_dim]), z1], axis=1)

            # fully connected layers
            output_dim = np.prod(outputs.get_shape().as_list()[1:])
            outputs = tf.reshape(outputs, [-1, output_dim])

            for i, hu in enumerate(self._model_conf["hu_z2_enc"]):
                outputs = fully_connected(inputs=outputs,
                                          num_outputs=hu,
                                          activation_fn=nn.relu,
                                          normalizer_fn=normalizer_fn,
                                          normalizer_params=normalizer_params,
                                          weights_regularizer=weights_regularizer,
                                          reuse=reuse,
                                          scope="z2_enc_fc%s" % (i + 1))

            z2_mu, z2_logvar, z2 = dense_latent(
                    outputs, self._model_conf["n_latent2"],
                    logvar_nl=self._model_conf["z2_logvar_nl"],
                    reuse=reuse, scope="z2_enc_lat")
        
        return [z2_mu, z2_logvar], z2

    def _build_rnn_decoder_and_recon_x(
            self, inputs, targets, training, reuse=False):
        with tf.variable_scope("dec_rec_and_recon_x", reuse=reuse):
            C, T, F = self._model_conf["target_shape"]

            Cell = _cell_dict[self._model_conf["rec_cell_type"]]
            cell = MultiRNNCell([Cell(hu) \
                    for hu in self._model_conf["rec_dec"]])

            if self._model_conf["rec_learn_init"]:
                raise NotImplementedError
            else:
                input_shape = tuple(array_ops.shape(input_) \
                        for input_ in nest.flatten(inputs))
                batch_size = input_shape[0][0]
                init_state = cell.zero_state(
                        batch_size, self._model_conf["input_dtype"])

            rec_dec_inp = self._model_conf["rec_dec_inp_test"]
            if training:
                rec_dec_inp = self._model_conf["rec_dec_inp_train"]

            if rec_dec_inp is not None:
                n_concur = self._model_conf["rec_dec_concur"]
                if T % n_concur != 0:
                    raise ValueError("total time steps must be " + \
                            "multiples of rec_dec_concur")
                n_frame = T // n_concur
            else:
                n_frame = T
            n_hist = self._model_conf["rec_dec_inp_hist"]
            info("decoder: n_frame=%s, n_concur=%s, n_hist=%s" % (
                    n_frame, n_concur, n_hist))

            def make_hist(hist, new_hist):
                with tf.name_scope("make_hist"):
                    if not self._model_conf["x_conti"]:
                        # TODO add target embedding?
                        new_hist = tf.cast(new_hist, tf.float32)

                    if n_hist > n_concur:
                        diff = n_hist - n_concur
                        return tf.concat(
                                [hist[:, :, -diff:, :], new_hist], 
                                axis=-2)
                    else:
                        return new_hist[:, :, -n_hist:, :]

            outputs = []
            if self._model_conf["x_conti"]:
                x_mu, x_logvar, x = [], [], []
            else:
                x_logits, x = [], []
            state_f = init_state
            hist = tf.zeros(
                    (array_ops.shape(inputs)[0], C, n_hist, F),
                    dtype=self._model_conf["input_dtype"],
                    name="init_hist")

            for f in xrange(n_frame):
                input_f = inputs
                if rec_dec_inp:
                    input_f = tf.concat(
                            [inputs, tf.reshape(hist, (-1, C * n_hist * F))],
                            axis=-1,
                            name="input_f_%s" % f)
                if f > 0:
                    tf.get_variable_scope().reuse_variables()

                output_f, state_f = cell(input_f, state_f)
                outputs.append(output_f)
                
                # TODO: input hist as well (like sampleRNN)?
                if self._model_conf["x_conti"]:
                    x_mu_f, x_logvar_f, x_f = dense_latent(
                            inputs=output_f, 
                            num_outputs=C * n_concur * F, 
                            mu_nl=self._model_conf["x_mu_nl"],
                            logvar_nl=self._model_conf["x_logvar_nl"],
                            scope="recon_x_f")
                    x_mu.append(tf.reshape(
                        x_mu_f, (-1, C, n_concur, F), 
                        name="recon_x_mu_f_4d"))
                    x_logvar.append(tf.reshape(
                        x_logvar_f, (-1, C, n_concur, F),
                        name="recon_x_logvar_f_4d"))
                    x.append(tf.reshape(
                        x_f, (-1, C, n_concur, F),
                        name="recon_x_f_4d"))
                    
                    if rec_dec_inp == "targets":
                        t_slice = slice(f * n_concur, (f + 1) * n_concur)
                        hist = make_hist(hist, targets[:, :, t_slice, :])
                    elif rec_dec_inp == "x_mu":
                        hist = make_hist(hist, x_mu[-1])
                    elif rec_dec_inp == "x":
                        hist = make_hist(hist, x[-1])
                    elif rec_dec_inp:
                        raise ValueError("unsupported rec_dec_inp (%s)" % (
                                rec_dec_inp))
                else:
                    raise ValueError
                    # n_bins = self._model_conf["n_bins"]
                    # x_logits_f, x_f = cat_dense_latent(
                    #         inputs=output_f, 
                    #         num_outputs=C * n_concur * F, 
                    #         n_bins=n_bins,
                    #         scope="recon_x_f")
                    # x_logits.append(tf.reshape(
                    #         x_logits_f, 
                    #         (-1, C, n_concur, F, n_bins),
                    #         name="recon_x_logits_f_5d"))
                    # x.append(tf.reshape(
                    #         x_f, 
                    #         (-1, C, n_concur, F),
                    #         name="recon_x_f_4d"))

                    # if rec_dec_inp == "targets":
                    #     t_slice = slice(f * n_concur, (f + 1) * n_concur)
                    #     hist = make_hist(hist, targets[:, :, t_slice, :])
                    # elif rec_dec_inp == "x_max":
                    #     hist = make_hist(hist, tf.argmax(x_logits[-1], -1))
                    # elif rec_dec_inp == "x":
                    #     hist = make_hist(hist, x[-1])
                    # elif rec_dec_inp:
                    #     raise ValueError("unsupported rec_dec_inp (%s)" % (
                    #             rec_dec_inp))
            
            # (bs, n_frame, top_rnn_hu)
            outputs = tf.stack(outputs, axis=1, name="rec_outputs")
            x = tf.concat(x, axis=2, name="recon_x_t_4d")
            
            if self._model_conf["x_conti"]:
                x_mu = tf.concat(x_mu, axis=2, name="recon_x_mu_t_4d")
                x_logvar = tf.concat(x_logvar, axis=2, name="recon_x_logvar_t_4d")
                px = [x_mu, x_logvar]
            else:
                x_logits = tf.concat(x_logits, axis=2, name="recon_x_logits_t_5d")
                px = x_logits

        return outputs, px, x

    def _build_decoder(self, z1, z2, reuse=False):
        # consider include ``target'' into args, 
        # since it may be used during training
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
                                          scope="dec_fc%s" % (i + 1))
            
            # if no recurrent layers, use dense_latent for target
            if not self._model_conf["rec_dec"]:
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
                    raise ValueError
                    # n_bins = self._model_conf["n_bins"]
                    # x_logits, x = cat_dense_latent(
                    #         outputs, target_dim, n_bins, reuse=reuse, scope="dec_lat")
                    # x_logits = tf.reshape(x_logits, [-1] + target_shape + [n_bins])
                    # px = x_logits

                x = tf.reshape(x, [-1] + target_shape)
            else:
                targets = None 
                if self.training:
                    targets = self._feed_dict["targets"]
                outputs, px, x = self._build_rnn_decoder_and_recon_x(
                        outputs, targets, self.training, reuse)


        return px, x
