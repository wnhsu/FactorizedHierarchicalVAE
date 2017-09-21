"""Base Factorized Hierarchical Variational Autoencoder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import array_ops
from collections import OrderedDict
from models import *
from libs.costs import kld, log_gauss, log_normal
from libs.activations import custom_relu
sce_logits = tf.nn.sparse_softmax_cross_entropy_with_logits

class BaseFacHierVAE(object):
    """
    Abstract class for factorized hierarchical VAE, 
    should never be called directly.
    """
    def __init__(self, model_conf, train_conf):
        # create data members
        self._model_conf    = None
        self._train_conf    = None

        self._feed_dict     = None  # feed dict needed for outputs
        self._layers        = None  # outputs at each layer
        self._outputs       = None  # general outputs (acc, posterior...)
        self._global_step   = None  # global_step for saver

        self._ops           = None  # accessible ops (train_step, decay_op...)

        # set model conf
        self._set_model_conf(model_conf)

        # set train conf
        self._set_train_conf(train_conf)

        # build model
        self._build_model()

        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        debug("Created saver for these variables:\n%s" % str(
                [p.name for p in tf.global_variables()]))
    
    @property
    def model_conf(self):
        return self._model_conf

    @property
    def train_conf(self):
        return self._train_conf

    @property
    def feed_dict(self):
        return self._feed_dict

    @property
    def outputs(self):
        return self._outputs

    @property
    def grads(self):
        return self._grads

    @property
    def global_step(self):
        return self._global_step

    @property
    def ops(self):
        return self._ops

    def _set_model_conf(self, model_conf):
        self._model_conf = {"input_shape": None,
                            "input_dtype": tf.float32,
                            "target_shape": None,
                            "target_dtype": tf.float32,
                            "n_latent1": 64,
                            "n_latent2": 64,
                            "n_class1": None,
                            "latent1_std": 0.5,
                            "x_conti": True,
                            "x_mu_nl": None,
                            "x_logvar_nl": None,
                            "n_bins": None}
        raise NotImplementedError

    def _set_train_conf(self, train_conf):
        self._train_conf = {"lr": 0.001,
                            "lr_decay_factor": 0.8,
                            "l2_weight": 0.0001,
                            "alpha_dis": 1,
                            "max_grad_norm": None,
                            "opt": "adam",
                            "opt_opts": {}}
        for k in train_conf:
            if k in self._train_conf:
                self._train_conf[k] = train_conf[k]
            else:
                info("WARNING: unused train_conf: %s" % str(k))
        
        info("=" * 20)
        info("TRAIN CONF:")
        for k, v in self._train_conf.items():
            info("%s: %s" % (k, v))
        info("=" * 20)
        
        for k in ["lr", "lr_decay_factor", "l2_weight", "max_grad_norm"]:
            if self._train_conf[k] is not None:
                self._train_conf[k] = tf.get_variable(
                        k, trainable=False, initializer=self._train_conf[k])

    def _build_model(self):
        # create placeholders
        inputs = tf.placeholder(
                self._model_conf["input_dtype"],
                shape=(None,)+self._model_conf["input_shape"],
                name="inputs")
        targets = tf.placeholder(
                self._model_conf["target_dtype"],
                shape=(None,)+self._model_conf["target_shape"],
                name="targets")
        labels = tf.placeholder(
                tf.int64, 
                shape=(None,),
                name="labels")
        N = tf.placeholder(tf.float32, shape=(None,), name="N")
        masks = tf.placeholder(
                tf.float32,
                shape=(None,)+self._model_conf["target_shape"],
                name="masks")

        is_train = tf.placeholder(tf.bool, name="is_train")
        self._feed_dict = {"inputs": inputs,
                           "targets": targets,
                           "labels": labels,
                           "N": N,
                           "masks": masks,
                           "is_train": is_train}

        # build {z1,z2}_encoder/decoder and outputs
        # p/q are distribution paramters, 
        #   [mu, logvar] for gaussian, 
        #   logits of shape x.shape + (n_bins,) for discrete

        qz1_x, sampled_z1 = self._build_z1_encoder(inputs)
        mu1_table, mu1 = self._build_mu1_lookup(labels)
        qz2_x, sampled_z2 = self._build_z2_encoder(inputs, sampled_z1)
        px_z, sampled_x = self._build_decoder(sampled_z1, sampled_z2)

        with tf.name_scope("costs"):
            # labeled data costs
            with tf.name_scope("log_pmu1"):
                log_pmu1 = tf.reduce_sum(log_normal(mu1), axis=1)
            
            with tf.name_scope("neg_kld_z1"):
                logvar1 = tf.ones_like(mu1) * \
                        np.log(np.power(self._model_conf["latent1_std"], 2))
                info("logvar of z1 given mu1 is %s" % (
                        np.log(np.power(self._model_conf["latent1_std"], 2))))
                pz1 = [mu1, logvar1]
                neg_kld_z1 = tf.reduce_sum(-1 * kld(*(qz1_x + pz1)), axis=1)

            with tf.name_scope("neg_kld_z2"):
                neg_kld_z2 = tf.reduce_sum(-1 * kld(*qz2_x), axis=1)

            with tf.name_scope("logpx_z"):
                masks = self._feed_dict["masks"]
                targets = self._feed_dict["targets"]
                if self._model_conf["x_conti"]:
                    info("p(x|z1,z2) is gaussian")
                    px_mu, px_logvar = px_z
                    logpx_z = tf.reduce_sum(
                            masks * log_gauss(px_mu, px_logvar, targets),
                            axis=(1, 2, 3))
                else:
                    info("p(x|z1,z2) is softmax")
                    sce = sce_logits(labels=targets, logits=px_z)
                    logpx_z = tf.reduce_sum(
                            masks * -1 * sce,
                            axis=(1, 2, 3))
            
            latent1_var = tf.pow(self._model_conf["latent1_std"], 2, name="latent1_var")

            with tf.name_scope("log_qy1"):
                # -(z2 - z2_mu2)^2/z2_var
                logits = tf.expand_dims(qz1_x[0], 1) - tf.expand_dims(mu1_table, 0)
                logits = -tf.pow(logits, 2) / (2 * latent1_var)
                logits = tf.reduce_sum(logits, axis=-1, name="qy1_logits")
                log_qy1 = -sce_logits(labels=labels, logits=logits)

            with tf.name_scope("lb"):
                # original variational lower bound for labeled data
                lb = logpx_z + neg_kld_z2 + neg_kld_z1 + (log_pmu1 / N)

            with tf.name_scope("lb_alpha"):
                # combine an additional weighted loss term
                alpha = self._train_conf["alpha_dis"]
                if alpha == 0.0:
                    lb_alpha = lb
                    info("use non-discriminative training")
                else:
                    lb_alpha = lb + alpha * log_qy1
                    info("use discriminative training")

            # L2 regularization
            with tf.name_scope("reg_loss"):
                reg_loss = tf.reduce_sum(tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES))

            # total loss
            with tf.name_scope("total_loss"):
                lb_loss = -tf.reduce_mean(lb_alpha)
                total_loss = reg_loss + lb_loss

        self._outputs = {
                # model params
                "mu1_table": mu1_table,
                # random variables/distributions
                "mu1": mu1,
                "qz1_x": qz1_x,
                "qz2_x": qz2_x,
                "px_z": px_z,
                "sampled_z1": sampled_z1,
                "sampled_z2": sampled_z2,
                "sampled_x": sampled_x,
                # costs
                "log_pmu1": log_pmu1,
                "neg_kld_z1": neg_kld_z1,
                "neg_kld_z2": neg_kld_z2,
                "logpx_z": logpx_z,
                "log_qy1": log_qy1,
                "lb": lb,
                "lb_alpha": lb_alpha,
                "reg_loss": reg_loss,
                "total_loss": total_loss
                }
        
        # create ops for training
        self._build_train()
        
    def _build_train(self):
        # create grads and clip optionally
        params = tf.trainable_variables()
        self._global_step = tf.get_variable(
                "global_step", trainable=False, initializer=0.0)
        
        with tf.name_scope("grad"):
            grads = tf.gradients(self._outputs["total_loss"], params)
            if self._train_conf["max_grad_norm"] is None:
                clipped_grads = grads
            else:
                clipped_grads, _ = tf.clip_by_global_norm(
                        grads, self._train_conf["max_grad_norm"])

        debug("GRADIENTS:")
        for i, param in enumerate(params):
            debug("%s: %s" % (param, clipped_grads[i]))
        self._grads = OrderedDict(
                zip(["grad_%s" % param.name for param in params], clipped_grads))

        # create ops
        with tf.name_scope("train"):
            lr = self._train_conf["lr"]
            opt = self._train_conf["opt"]
            opt_opts = self._train_conf["opt_opts"]

            if opt == "adam":
                info("Using Adam as the optimizer")
                opt = tf.train.AdamOptimizer(learning_rate=lr, **opt_opts)
            elif opt == "sgd":
                info("Using SGD as the optimizer")
                opt = tf.train.GradientDescentOptimizer(learning_rate=lr, **opt_opts)
            else:
                raise ValueError("optimizer %s not supported" % opt)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = opt.apply_gradients(zip(clipped_grads, params),
                                                 global_step=self._global_step)
            
            decay_op = lr.assign(lr * self._train_conf["lr_decay_factor"])

        self._ops = {"train_step": train_step, "decay_op": decay_op}
        
    def _build_z1_encoder(self, inputs, reuse=False):
        """return q(z1 | x), sampled_z1"""
        raise NotImplementedError

    def _build_mu1_lookup(self, labels, reuse=False):
        n_class1 = self._model_conf["n_class1"]
        n_latent1 = self._model_conf["n_latent1"]
        with tf.variable_scope("mu1", reuse=reuse):
            with tf.device("/cpu:0"):
                mu1_table = tf.get_variable(
                        name="mu1_table", 
                        trainable=True,
                        initializer=tf.random_normal([n_class1, n_latent1]))
                mu1 = tf.gather(mu1_table, labels)
        return mu1_table, mu1

    def _build_z2_encoder(self, inputs, z1, reuse=False):
        """return q(z2 | x, z1), sampled_z2"""
        raise NotImplementedError

    def _build_decoder(self, z1, z2, reuse=False):
        """return p(x | z1, z2), sampled_x"""
        raise NotImplementedError

    def init_or_restore_model(self, sess, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            info("Reading model params from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            info("Creating model with fresh params")
            sess.run(tf.global_variables_initializer())
        return sess.run(self._global_step)
