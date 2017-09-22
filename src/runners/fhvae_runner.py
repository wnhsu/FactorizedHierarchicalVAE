"""Template runner for training and testing Factorized Hierarchical VAE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import os
import time
import shutil
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict, defaultdict

from utils import *
from runners import *
from tools.vis import plot_rows, plot_grids
from tools.audio import convert_to_complex_spec, complex_spec_to_audio
from tools.analysis import pca

from kaldi_io import BaseFloatMatrixWriter as BFMWriter
from kaldi_io import BaseFloatVectorWriter as BFVWriter
from kaldi_io import SequentialBaseFloatVectorReader as SBFVReader

DEFAULT_WRITE_FN = lambda uttid, feats: print("%s\n%s" % (uttid, feats))
DEFAULT_DUMP_LATENT_OPT = {
        "use_z1": False,
        "use_z2": True,
        "use_mean": True,
        "use_logvar": True,
        "utt_left_pad": 0,
        "utt_right_pad": 0,}
DEFAULT_DUMP_REPR_OPT = {
        "which_repr": "mu1",}   # "model_mu1/mu1/mu2"
DEFAULT_REPL_REPR_OPT = {}
DEFAULT_VIS_FAC_OPT = {
        "n": 1,                 # number of segments drawn from each utt in *_utt_list
        "z1_utt_list": None,    # list of utterance for z1
        "z2_utt_list": None,    # list of utterance for z2
        "z1_segs": None,        # ignore `n` and `z1_utt_list`, segments for z1
        "z2_segs": None,        # ignore `n` and `z2_utt_list`, segments for z2
        }
DEFAULT_TRAV_OPT = {
        "n": 5,                 # number of seed segments
        "seed_utt_list": None,  # ignore `n`, draw 1 segment from each utt
        "seed_segs": None,      # ignore `n` and `seed_utt`, specify which segs to traverse
        "k": 9,                 # num_intervals
        "trav_range": (-3, 3),  # range of traversing values
        "figsize": (10, 30),
        "feat_type": "spec",
        }

def train(exp_dir, set_name, model, train_set, dev_set, 
        train_conf, n_print_steps=200, debug=False):
    if os.path.exists("%s/.done" % exp_dir):
        info("training is already done. exit..")
        return

    dev_iterator_fn     = lambda: dev_set.iterator(2048, set_name) if dev_set else None 
    train_label_to_N    = train_set.get_label_N(set_name)
    n_class             = train_set.get_n_class(set_name)
    dev_label_to_N      = dev_set.get_label_N(set_name)
    n_epochs            = train_conf.pop("n_epochs")
    n_patience          = train_conf.pop("n_patience")
    bs                  = train_conf.pop("bs")
    n_steps_per_epoch   = train_conf.pop("n_steps_per_epoch")
    latent1_var         = np.power(model.model_conf["latent1_std"], 2)
    assert(n_steps_per_epoch > 0)

    model_dir = "%s/models" % exp_dir
    check_and_makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")
    
    # create summaries
    sum_names = ["lb", "logpx_z", "log_pmu1", "neg_kld_z1", "neg_kld_z2", "log_qy1"]
    sum_vars = [tf.reduce_mean(model.outputs[name]) for name in sum_names]
    with tf.variable_scope("train"):
        train_summaries = tf.summary.merge(
                [tf.summary.scalar(*p) for p in zip(sum_names, sum_vars)])

    test_sum_names = ["lb", "logpx_z", "log_pmu1", "neg_kld_z1", "neg_kld_z2"]
    test_sum_vars = [tf.reduce_mean(model.outputs[name]) for name in test_sum_names]
    with tf.variable_scope("test"):
        test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in test_sum_names])
        test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, test_vars[k]) for k in test_vars])

    def make_feed_dict(inputs, targets, labels, N, is_train):
        feed_dict = {model.feed_dict["labels"]: labels,
                     model.feed_dict["inputs"]: inputs,
                     model.feed_dict["targets"]: targets,
                     model.feed_dict["N"]: N,
                     model.feed_dict["masks"]: np.ones_like(inputs),
                     model.feed_dict["is_train"]: is_train}
        return feed_dict

    global_step = -1
    epoch = -1
    passes = 0     # number of dataset passes this run
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        init_step = model.init_or_restore_model(sess, model_dir)
        global_step = int(init_step)
        epoch = int(global_step // n_steps_per_epoch)
        info("init or restore model takes %.2f s" % (time.time() - start_time))
        info("current #steps=%s, #epochs=%s" % (global_step, epoch))
        if epoch >= n_epochs:
            info("training is already done. exit..")
            return
        train_writer = tf.summary.FileWriter("%s/log/train" % exp_dir, sess.graph)
        dev_writer = tf.summary.FileWriter("%s/log/dev" % exp_dir)
        
        info("start training...")
        best_epoch = -1
        best_dev_lb = -np.inf
        train_start_time = time.time()
        print_start_time = time.time()

        while True:
            for inputs, _, labels, targets in train_set.iterator(bs, set_name):
                N = _make_N(train_label_to_N, labels)
                if debug:
                    outputs, global_step, _ = sess.run(
                            [sum_vars, model.global_step, model.ops["train_step"]],
                            make_feed_dict(inputs, targets, labels, N, 1))
                    info("[epoch %.f step %.f pass %.f]: " % (
                                epoch, global_step, passes) + \
                            ", total time=%.2fs, " % (
                                time.time() - train_start_time) + \
                            ", ".join(["%s %.4f" % p for p in zip(
                                sum_names, outputs)]))
                else:
                    global_step, _ = sess.run(
                            [model.global_step, model.ops["train_step"]],
                            make_feed_dict(inputs, targets, labels, N, 1))

                if global_step % n_print_steps == 0 and global_step != init_step:
                    outputs, summary = sess.run(
                            [sum_vars, train_summaries],
                            make_feed_dict(inputs, targets, labels, N, 0))
                    train_writer.add_summary(summary, global_step)
                    info("[epoch %.f step %.f pass %.f]: " % (
                                epoch, global_step, passes) + \
                            "print time=%.2fs" % (
                                time.time() - print_start_time) + \
                            ", total time=%.2fs, " % (
                                time.time() - train_start_time) + \
                            ", ".join(["%s %.4f" % p for p in zip(
                                sum_names, outputs)]))
                    print_start_time = time.time()
                    if np.isnan(outputs[0]):
                        info("...exit training and not saving this epoch")
                        return

                if global_step % n_steps_per_epoch == 0 and global_step != init_step:
                    if dev_iterator_fn:
                        val_start_time = time.time()
                        dev_vals = _valid(
                                sess, model, test_sum_names, test_sum_vars,
                                dev_label_to_N, latent1_var, dev_iterator_fn)
                        feed_dict = dict(zip(test_vars.values(), dev_vals.values()))
                        summary = sess.run(test_summaries, feed_dict)
                        dev_writer.add_summary(summary, global_step)
                        info("[epoch %.f]: dev  \t" % epoch + \
                                "valid time=%.2fs" % (
                                    time.time() - val_start_time) + \
                                ", total time=%.2fs, " % (
                                    time.time() - train_start_time) + \
                                ", ".join(["%s %.4f" % p for p in dev_vals.items()]))
                        if dev_vals["lb"] > best_dev_lb:
                            best_epoch, best_dev_lb = epoch, dev_vals["lb"]
                            model.saver.save(sess, ckpt_path)
                        if epoch - best_epoch > n_patience:
                            info("...running out of patience" + \
                                    ", time elapsed=%.2fs" % (
                                        time.time() - train_start_time) + \
                                    ", exit training")
                            open("%s/.done" % exp_dir, "a")
                            return

                    epoch += 1
                    if epoch >= n_epochs:
                        info("...finish training" + \
                                ", time elapsed=%.2fs" % (
                                    time.time() - train_start_time))
                        open("%s/.done" % exp_dir, "a")
                        return

            passes += 1

def test(exp_dir, set_name, model, test_set):
    test_iterator_fn    = lambda: test_set.iterator(2048, set_name)
    test_label_to_N     = test_set.get_label_N(set_name)
    latent1_var         = np.power(model.model_conf["latent1_std"], 2)

    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")

    # create summaries
    sum_names = ["lb", "logpx_z", "log_pmu1", "neg_kld_z1", "neg_kld_z2"]
    sum_vars = [tf.reduce_mean(model.outputs[name]) for name in sum_names]
    
    with tf.variable_scope("test"):
        test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in sum_names])
        test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, test_vars[k]) for k in test_vars])

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        test_writer = tf.summary.FileWriter("%s/log/test" % exp_dir)

        test_vals = _valid(
                sess, model, sum_names, sum_vars, test_label_to_N, 
                latent1_var, test_iterator_fn, debug=False)
        feed_dict = dict(zip(test_vars.values(), test_vals.values()))
        summary, global_step = sess.run([test_summaries, model.global_step], feed_dict)
        test_writer.add_summary(summary, global_step)
        info("test\t" + ", ".join(["%s %.4f" % p for p in test_vals.items()]))

def dump_latent(
        exp_dir, model, dataset, label2str,
        write_fn=DEFAULT_WRITE_FN, opts=DEFAULT_DUMP_LATENT_OPT):
    iterator_fn_by_label = \
            lambda label: dataset.iterator_by_label(2048, "uttid", label)
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")

    output_names = []
    if opts["use_z1"]:
        output_names.append("qz1_x")
    if opts["use_z2"]:
        output_names.append("qz2_x")

    output_vars = []
    for name in output_names:
        if opts["use_mean"]:
            output_vars.append(model.outputs[name][0])
        if opts["use_logvar"]:
            output_vars.append(model.outputs[name][1])
    output_var = tf.concat(output_vars, axis=-1)

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))

        utt_id_list = dataset.sample_utt_id(-1)
        for label in [dataset.get_label("uttid", utt_id) for utt_id in utt_id_list]:
            feats = []
            for inputs, _, _, _ in iterator_fn_by_label(label):
                batch_feats = sess.run(
                        output_var,
                        feed_dict={
                            model.feed_dict["inputs"]: inputs,
                            model.feed_dict["is_train"]: 0})
                feats.append(batch_feats)
            feats = np.concatenate(feats, axis=0)
            feats = _pad_feats(feats, opts["utt_left_pad"], opts["utt_right_pad"])
            debug("writing %s, #frames=%s, dim=%s" % (label2str[label], len(feats), feats.shape[1]))
            write_fn(label2str[label], feats)

    info("...finish dumping latent, time elapsed=%.2fs" % (time.time() - start_time))

def dump_repr(
        exp_dir, model, dataset, label2str, repr_set_name,
        write_fn=DEFAULT_WRITE_FN, opts=DEFAULT_DUMP_REPR_OPT):
    iterator_fn     = lambda: dataset.iterator(2048, repr_set_name)
    label_to_N      = dataset.get_label_N(repr_set_name)
    latent1_var     = np.power(model.model_conf["latent1_std"], 2)

    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        
        if opts["which_repr"] == "model_mu1":
            label2repr = sess.run(model.outputs["mu1_table"], feed_dict={})
        elif opts["which_repr"] == "mu1":
            label2repr = _est_mu1(sess, model, label_to_N, latent1_var, iterator_fn, False)
        elif opts["which_repr"] == "mu2":
            label2repr = _est_mu2(sess, model, label_to_N, latent1_var, iterator_fn, False)

        for label in sorted(label2repr.keys()):
            debug("writing %s" % label2str[label])
            write_fn(label2str[label], label2repr[label])

        pca(np.array(label2repr.values()))
    info("...finish dumping representations, time elapsed=%.2fs" % (time.time() - start_time))

def repl_repr_utt(
        exp_dir, model, dataset, repr_set_name, label_to_str,
        label_str_to_repr, repl_list, write_fn=DEFAULT_WRITE_FN, 
        opts=DEFAULT_REPL_REPR_OPT):
    """modify the representations of selected labels
    feat_rspec set to test_feat_rspec by default
    label for mod_label_in is repr_label (str)"""
    iterator_by_utt_str_fn = \
            lambda utt_str: dataset.iterator_by_label(
                    2048, "uttid", dataset.get_label("uttid", utt_str))
    
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")
  
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        
        start_time = time.time()
        for i, (utt_str, label_str) in enumerate(repl_list):
            utt_feats = np.concatenate(
                    [inputs for inputs, _, _, _ in iterator_by_utt_str_fn(utt_str)],
                    axis=0)
            src_label_str = label_to_str[dataset.get_label(repr_set_name, utt_str)]
            src_repr = label_str_to_repr[src_label_str]
            tar_repr = label_str_to_repr[label_str]
            debug("utt: %s, shape %s" % (utt_str, utt_feats.shape) + \
                    ", %s label: %s => %s" % (repr_set_name, src_label_str, label_str))
            Z1, Z1_mod, Z2, utt_feats_mod = _replace_repr(
                    sess, model, utt_feats, src_repr, tar_repr)
            raw_utt_feats_mod = dataset.undo_mvn(dataset.target_to_feat(utt_feats_mod))
            utt_str_mod = "%s_modto_%s" % (utt_str, label_str)
            
            # NOTES: except for the first segment, use only last frame
            raw_utt_segs_mod = [raw_utt_feats_mod[0]]    # (C, seg_len, F)
            for raw_utt_seg_mod in raw_utt_feats_mod[1:]:
                raw_utt_segs_mod.append(raw_utt_seg_mod[:, -1:, :])
            nonoverlap_raw_utt_feats_mod = np.concatenate(raw_utt_segs_mod, axis=-2)
            write_fn(utt_str_mod, nonoverlap_raw_utt_feats_mod)

    info("...finish replacing representations, time elapsed=%.2fs" % (time.time() - start_time))

def factorize(exp_dir, model, dataset, write_fn=DEFAULT_WRITE_FN, opts=DEFAULT_VIS_FAC_OPT):
    """
    visualize latent variable factorization
    """
    assert(opts["z1_utt_list"] is not None or opts["z1_segs"] is not None)
    assert(opts["z2_utt_list"] is not None or opts["z2_segs"] is not None)

    iterator_by_utt_str_fn = \
            lambda utt_str: dataset.iterator_by_label(
                    2048, "uttid", dataset.get_label("uttid", utt_str))
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))

        # X1, X2 shape is (n*len(z_utt_list)|len(z_segs),) + feat_shape
        np.random.seed(111)
        if opts["z1_segs"]:
            X1, _, _, _ = dataset.get_item_by_segs(opts["z1_segs"])
            X1_utt_id_list = [seg.utt_id for seg in opts["z1_segs"]]
        else:
            X1, _, _, _ = dataset.sample_item(opts["z1_utt_list"], opts["n"])
            X1_utt_id_list = [utt_id for utt_id in opts["z1_utt_list"] for _ in xrange(opts["n"])]

        if opts["z2_segs"]:
            X2, _, _, _ = dataset.get_item_by_segs(opts["z2_segs"])
            X2_utt_id_list = [seg.utt_id for seg in opts["z2_segs"]]
        else:
            X2, _, _, _ = dataset.sample_item(opts["z2_utt_list"], opts["n"])
            X2_utt_id_list = [utt_id for utt_id in opts["z2_utt_list"] for _ in xrange(opts["n"])]

        Z1 = _encode_z1_mean_fn(sess, model, X1)
        Z2 = _encode_z2_mean_fn(sess, model, X2)

        # X_fac[i,j] is combining i-th Z1 and j-th Z2
        X_fac = np.array([[_decode_x_mean_fn(sess, model, [z1], [z2])[0] for z2 in Z2] for z1 in Z1])
        raw_X_fac = np.array([dataset.undo_mvn(X_fac_row) for X_fac_row in X_fac])
        raw_X1 = dataset.undo_mvn(_decode_x_mean_fn(sess, model, *_sample_z1_z2_fn(sess, model, X1)))
        raw_X2 = dataset.undo_mvn(_decode_x_mean_fn(sess, model, *_sample_z1_z2_fn(sess, model, X2)))
        raw_X1_ori = dataset.undo_mvn(X1)
        raw_X2_ori = dataset.undo_mvn(X2)
        
        for i, (raw_X_fac_row, X1_utt_id) in enumerate(zip(raw_X_fac, X1_utt_id_list)):
            for j, (raw_x_fac, X2_utt_id) in enumerate(zip(raw_X_fac_row, X2_utt_id_list)):
                utt_str_fac = "%s_%s_%s_FACSEP_%s" % (i, j, X1_utt_id, X2_utt_id)
                write_fn(utt_str_fac, raw_x_fac)
        for i, (raw_x1, raw_x1_ori, X1_utt_id) in enumerate(zip(raw_X1, raw_X1_ori, X1_utt_id_list)):
            write_fn("%s_-1_%s_ori" % (i, X1_utt_id), raw_x1_ori)
            write_fn("%s_-1_%s" % (i, X1_utt_id), raw_x1)
        for i, (raw_x2, raw_x2_ori, X2_utt_id) in enumerate(zip(raw_X2, raw_X2_ori, X2_utt_id_list)):
            write_fn("-1_%s_%s_ori" % (i, X2_utt_id), raw_x2_ori)
            write_fn("-1_%s_%s" % (i, X2_utt_id), raw_x2)

def traverse(exp_dir, model, dataset, img_dir, opts=DEFAULT_TRAV_OPT):
    iterator_fn     = lambda: dataset.iterator(2048)
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "fhvae.ckpt")

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))

        np.random.seed(0)
        if opts["seed_segs"]:
            seed_data, _, _, _ = dataset.get_item_by_segs(opts["seed_segs"])
        elif opts["seed_utt_list"]:
            seed_data, _, _, _ = dataset.sample_item(opts["seed_utt_list"], 1)
        else:
            seed_utt = dataset.sample_utt_id(opts["n"])
            seed_data, _, _, _ = dataset.sample_item(seed_utt, 1)

        trav_z1_grids, trav_z2_grids = _traverse(
                sess, model, seed_data, opts["k"], opts["trav_range"])

    check_and_makedirs(img_dir)
    for d, trav_z1_grid in enumerate(trav_z1_grids):
        trav_z1_grid = np.array([dataset.undo_mvn(trav_z1) for trav_z1 in trav_z1_grid])
        plot_grids(
                [trav_z1_grid[:, :1, ...], trav_z1_grid[:, 1:, ...]], 
                labels=["", "traversal of z1, dim=%s" % d],
                mode="save", name="%s/z1_%s.png" % (img_dir, d),
                feat_type=opts["feat_type"], figsize=opts["figsize"])

    for d, trav_z2_grid in enumerate(trav_z2_grids):
        trav_z2_grid = np.array([dataset.undo_mvn(trav_z2) for trav_z2 in trav_z2_grid])
        plot_grids(
                [trav_z2_grid[:, :1, ...], trav_z2_grid[:, 1:, ...]], 
                labels=["", "traversal of z2, dim=%s" % d],
                mode="save", name="%s/z2_%s.png" % (img_dir, d),
                feat_type=opts["feat_type"], figsize=opts["figsize"])

def _make_N(label_to_N, labels):
    assert(np.ndim(labels) == 1)
    return np.array([label_to_N[label] for label in labels])

def _make_mu1(label_to_mu1, labels):
    assert(np.ndim(labels) == 1)
    return np.array([label_to_mu1[label] for label in labels])

def _encode_z1_mean_fn(sess, model, inputs):
    feed_dict = {model.feed_dict["inputs"]: inputs,
                 model.feed_dict["is_train"]: 0}
    return sess.run(model.outputs["qz1_x"][0], feed_dict)

def _encode_z2_mean_fn(sess, model, inputs):
    """use z1 mean as sampled z1 for deterministic outputs"""
    Z1 = _encode_z1_mean_fn(sess, model, inputs)
    feed_dict = {model.feed_dict["inputs"]: inputs,
                 model.feed_dict["is_train"]: 0,
                 model.outputs["sampled_z1"]: Z1}
    return sess.run(model.outputs["qz2_x"][0], feed_dict)

def _decode_x_mean_fn(sess, model, Z1, Z2):
    feed_dict = {
            model.outputs["sampled_z1"]: Z1,
            model.outputs["sampled_z2"]: Z2,
            model.feed_dict["is_train"]: 0}
    return sess.run(model.outputs["px_z"][0], feed_dict)

def _sample_z1_z2_fn(sess, model, inputs):
    feed_dict = {model.feed_dict["inputs"]: inputs,
                 model.feed_dict["is_train"]: 0}
    outputs = [model.outputs["sampled_z1"], model.outputs["sampled_z2"]]
    return sess.run(outputs, feed_dict)

def _est_mu1(sess, model, label_to_N, latent1_var, iterator_fn, debug=False):
    """approximated MAP estimation of mu1 for dataset"""
    label_to_acc_z1 = defaultdict(float)
    label_to_mu1 = dict()
    n_batches = 0
    start_time = time.time()
    for inputs, _, labels, _ in iterator_fn():
        n_batches += 1
        Z1 = _encode_z1_mean_fn(sess, model, inputs)
        for label, z1 in zip(labels, Z1):
            label_to_acc_z1[label] += z1
        if debug and n_batches % 2000 == 0:
            info("_est_mu1: took %.2f(s) to process %s batches" % (
                time.time() - start_time, n_batches))
            start_time = time.time()
    acc_length = 0.0
    for label, acc_z1 in label_to_acc_z1.iteritems():
        label_to_mu1[label] = acc_z1 / (label_to_N[label] + latent1_var)
        length = np.linalg.norm(label_to_mu1[label])
        if debug:
            info("length=%.2f, N=%s, label=%s" % ( 
                length, label_to_N[label], label,))
        acc_length += length
    mean_length = acc_length / len(label_to_acc_z1)
    info("averaged length = %.2f, #labels = %s" % (mean_length, len(label_to_acc_z1)))
        
    return label_to_mu1

def _est_mu2(sess, model, label_to_N, latent1_var, iterator_fn, debug=False):
    """heuristic approximated MAP estimation of mu2 for dataset"""
    label_to_acc_z2 = defaultdict(float)
    label_to_mu2 = dict()
    n_batches = 0
    start_time = time.time()
    for inputs, _, labels, _ in iterator_fn():
        n_batches += 1
        Z2 = _encode_z2_mean_fn(sess, model, inputs)
        for label, z2 in zip(labels, Z2):
            label_to_acc_z2[label] += z2
    acc_length = 0.0
    for label, acc_z2 in label_to_acc_z2.iteritems():
        label_to_mu2[label] = acc_z2 / (label_to_N[label] + latent1_var)
        length = np.linalg.norm(label_to_mu2[label])
        if debug:
            info("length=%.2f, N=%s, label=%s" % (length, label_to_N[label], label,))
        acc_length += length
    mean_length = acc_length / len(label_to_acc_z2)
    info("averaged length = %.2f, #labels = %s" % (mean_length, len(label_to_acc_z2)))
        
    return label_to_mu2

def _valid(sess, model, sum_names, sum_vars, 
        label_to_N, latent1_var, iterator_fn, debug=False):
    label_to_mu1 = _est_mu1(
            sess, model, label_to_N, latent1_var, iterator_fn, debug)
    # valid_n_class1 = int(max(label_to_mu1.keys())) + 1
    # mu1_shape = label_to_mu1.values()[0].shape
    # mu1_dtype = label_to_mu1.values()[0].dtype
    # mu1_table = np.zeros([valid_n_class1] + list(mu1_shape), dtype=mu1_dtype)
    # for k in label_to_mu1:
    #     mu1_table[k] = label_to_mu1[k]

    vals = OrderedDict([(name, 0) for name in sum_names])
    n_batches = 0
    start_time = time.time()
    for inputs, _, labels, targets in iterator_fn():
        n_batches += 1
        N = _make_N(label_to_N, labels)
        mu1 = _make_mu1(label_to_mu1, labels)
        outputs = sess.run(
                sum_vars, 
                feed_dict={
                    # model.feed_dict["labels"]: labels,
                    # model.outputs["mu1_table"]: mu1_table,
                    model.feed_dict["inputs"]: inputs,
                    model.feed_dict["targets"]: targets,
                    model.feed_dict["N"]: N,
                    model.outputs["mu1"]: mu1,
                    model.feed_dict["masks"]: np.ones_like(inputs),
                    model.feed_dict["is_train"]: 0})
        for name, val in zip(sum_names, outputs):
            vals[name] += val
        if debug and n_batches % 2000 == 0:
            info("_est_mu1: took %.2f(s) to process %s batches" % (
                time.time() - start_time, n_batches))
            start_time = time.time()
    for name in vals:
        vals[name] /= n_batches
    return vals

def _traverse(sess, model, seed_data, k, trav_range):

    seed_z1s = sess.run(
            model.outputs["qz1_x"][0], 
            feed_dict={
                model.feed_dict["inputs"]: seed_data,
                model.feed_dict["is_train"]: 0})
    seed_z2s = sess.run(
            model.outputs["qz2_x"][0], 
            feed_dict={
                model.feed_dict["inputs"]: seed_data,
                model.outputs["sampled_z1"]: seed_z1s,
                model.feed_dict["is_train"]: 0})
    data_shape = list(seed_data.shape[1:])

    # trav_grids is indexed by latent variable dimension being traversed
    trav_vals = np.linspace(trav_range[0], trav_range[1], k)
    trav_z1_grids = [_traverse_dim(sess, model, seed_z1s, seed_z2s, data_shape, trav_vals, d, "z1") \
            for d in xrange(seed_z1s.shape[-1])]
    trav_z2_grids = [_traverse_dim(sess, model, seed_z1s, seed_z2s, data_shape, trav_vals, d, "z2") \
            for d in xrange(seed_z2s.shape[-1])]

    return trav_z1_grids, trav_z2_grids

def _traverse_dim(sess, model, seed_z1s, seed_z2s, data_shape, trav_vals, dim, which="z1"):
    n, k = len(seed_z1s), len(trav_vals)
    def gen_flat_trav_and_fixed_lats(seed_lats_to_trav, seed_lats_fixed):
        trav_lats = np.array([
            seed_lats_to_trav for _ in xrange(len(trav_vals) + 1)])
        trav_lats = trav_lats.transpose((1, 0, 2))  # shape=(n, k+1, n_latent)
        for i, val in enumerate(trav_vals):
            trav_lats[:, i + 1, dim] = val
        flat_trav_lats = trav_lats.reshape(n * (k + 1), -1)
        
        fixed_lats = np.array([
            seed_lats_fixed for _ in xrange(len(trav_vals) + 1)])
        fixed_lats = fixed_lats.transpose((1, 0, 2))
        flat_fixed_lats = fixed_lats.reshape(n * (k + 1), -1)

        return flat_trav_lats, flat_fixed_lats

    if which == "z1":
        flat_trav_z1s, flat_fixed_z2s = \
                gen_flat_trav_and_fixed_lats(seed_z1s, seed_z2s)
        flat_trav_data = sess.run(
                model.outputs["px_z"][0],
                feed_dict={
                    model.outputs["sampled_z1"]: flat_trav_z1s,
                    model.outputs["sampled_z2"]: flat_fixed_z2s,
                    model.feed_dict["is_train"]: 0})
    elif which == "z2":
        flat_trav_z2s, flat_fixed_z1s = \
                gen_flat_trav_and_fixed_lats(seed_z2s, seed_z1s)
        flat_trav_data = sess.run(
                model.outputs["px_z"][0],
                feed_dict={
                    model.outputs["sampled_z1"]: flat_fixed_z1s,
                    model.outputs["sampled_z2"]: flat_trav_z2s,
                    model.feed_dict["is_train"]: 0})
    else:
        raise ValueError("invalid which (%s)" % which)

    trav_data = flat_trav_data.reshape([n, k + 1] + data_shape)
    return trav_data

def _pad_feats(feats, left_pad, right_pad):
    """
    if feats is a (T, ...) matrix, 
    return a (left_pad + T +right_pad, ...) matrix
    """
    tile_rep_suffix = [1] * (np.asarray(feats).ndim - 1)
    feats = np.concatenate([
            np.tile(feats[0], [left_pad] + tile_rep_suffix),
            feats, 
            np.tile(feats[-1], [right_pad] + tile_rep_suffix)])
    return feats

def _replace_repr(sess, model, inputs, src_repr, tar_repr):
    Z1, Z2 = _sample_z1_z2_fn(sess, model, inputs)
    Z1_mod = Z1 - src_repr[np.newaxis, :] + tar_repr[np.newaxis, :]
    inputs_mod = _decode_x_mean_fn(sess, model, Z1_mod, Z2)
    return Z1, Z1_mod, Z2, inputs_mod
