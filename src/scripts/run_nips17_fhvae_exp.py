import os
import shutil
from runners import *
from runners.fhvae_runner import train, test, dump_latent, dump_repr, repl_repr_utt, factorize, traverse
from tools.kaldi.prep_kaldi_feat import flatten_channel
from datasets.segment import Segment
from datasets.datasets_loaders import datasets_loader
from datasets.datasets_loaders import get_frame_ra_dataset_conf, get_nonoverlap_ra_dataset_conf
from parsers.train_parsers import fhvae_train_parser as train_parser
from parsers.dataset_parsers import kaldi_ra_dataset_parser as dataset_parser
from tools.kaldi.plot_scp import plot_kaldi_feat
from tools.kaldi.plot_fac import plot_kaldi_feat_fac
from kaldi_io import BaseFloatMatrixWriter as BFMWriter
from kaldi_io import BaseFloatVectorWriter as BFVWriter
from kaldi_io import SequentialBaseFloatVectorReader as SBFVReader

DEFAULT_SET_NAME="uttid"
DEFAULT_TRAIN_CONF="conf/train/fhvae/e500_p50_lr1e-3_bs256_nbs2000_ad0.cfg"
DEFAULT_MODEL_CONF="conf/model/fhvae/lstm_2L_256_lat_32_32.cfg"

# flags for all actions
tf.app.flags.DEFINE_boolean("debug", False, "debugging mode (for training) if True")
tf.app.flags.DEFINE_string("arch", "rnn", "dnn|cnn|rnn")
tf.app.flags.DEFINE_string("exp_dir", "", "path to dump/load this experiment")

# flags for if running optional actions
tf.app.flags.DEFINE_boolean("test", True, "run testing if True")
tf.app.flags.DEFINE_boolean("dump_latent", False, "dump latent variable by frame if True")
tf.app.flags.DEFINE_boolean("dump_repr", False, "dump representations if True")
tf.app.flags.DEFINE_boolean("repl_repr_utt", False, "replace representations if True")
tf.app.flags.DEFINE_boolean("fac", False, "show factorization of latent variables if True")
tf.app.flags.DEFINE_boolean("trav", False, "traverse latent space")

# flags for train
tf.app.flags.DEFINE_string("set_name", DEFAULT_SET_NAME, "set name of label conditioned on")
tf.app.flags.DEFINE_string("train_conf", DEFAULT_TRAIN_CONF, "path to training config")
tf.app.flags.DEFINE_string("model_conf", DEFAULT_MODEL_CONF, "path to model config")
tf.app.flags.DEFINE_string("dataset_conf", "", "path to dataset config")
tf.app.flags.DEFINE_integer("n_print_steps", 200, "print training diagnostics every N steps")

tf.app.flags.DEFINE_float("alpha_dis", -1, "alpha for discriminative objective")
tf.app.flags.DEFINE_integer("n_latent1", 0, "z1 dimension")
tf.app.flags.DEFINE_integer("n_latent2", 0, "z2 dimension")

# flags for non-train actions
tf.app.flags.DEFINE_string("feat_rspec", "", "feat rspecifier to replace test_feat_rspec")
tf.app.flags.DEFINE_string("feat_set_name", "", "name(s) of label set(s)")
tf.app.flags.DEFINE_string("feat_utt2label_path", "", "utt2label_path(s) for feat_rspec")
tf.app.flags.DEFINE_string("feat_label_N", "", "number(s) of classes for feat_rspec")

# flags for dump_lat
tf.app.flags.DEFINE_string("train_lat_wspec", "", "train set write specifier")
tf.app.flags.DEFINE_string("dev_lat_wspec", "", "dev set write specifier")
tf.app.flags.DEFINE_string("test_lat_wspec", "", "test set write specifier")
tf.app.flags.DEFINE_string("train_utt_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_string("dev_utt_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_string("test_utt_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_boolean("use_z1", False, "dump z1")
tf.app.flags.DEFINE_boolean("use_z2", True, "dump z2")
tf.app.flags.DEFINE_boolean("use_mean", True, "dump mean of selected latent vars")
tf.app.flags.DEFINE_boolean("use_logvar", True, "dump logvar of selected latent vars")

# flags for dump_repr
tf.app.flags.DEFINE_string("train_repr_wspec", "", "train set write specifier")
tf.app.flags.DEFINE_string("dev_repr_wspec", "", "dev set write specifier")
tf.app.flags.DEFINE_string("test_repr_wspec", "", "test set write specifier")
tf.app.flags.DEFINE_string("train_repr_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_string("dev_repr_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_string("test_repr_id_map", "", "path to utterance id map")
tf.app.flags.DEFINE_string("repr_set_name", "", "set name of labels for feat_rspec")
tf.app.flags.DEFINE_string("repr_id_map", "", "path to label id map to dump")
tf.app.flags.DEFINE_string("which_repr", "mu1", "{model_mu1|mu1|mu2}. type of representations.")

# flags for repl_repr_utt
tf.app.flags.DEFINE_string("repl_utt_set_name", "", "set name for replacing representations")
tf.app.flags.DEFINE_string("repl_utt_repr_spec", "", "representation rspecifier")
tf.app.flags.DEFINE_string("repl_utt_list", "", "test set replacing representation list: (uttid_str, attr_str)")
tf.app.flags.DEFINE_string("repl_utt_wspec", "", "test set replacing representation write specifier")
tf.app.flags.DEFINE_string("repl_utt_img_dir", "", "test set replacing representation image directory")
tf.app.flags.DEFINE_string("repl_utt_id_map", "", "")

# flags for vis_fac
tf.app.flags.DEFINE_integer("fac_n", 1, "")
tf.app.flags.DEFINE_string("fac_z1_spec", "", "")
tf.app.flags.DEFINE_string("fac_z2_spec", "", "")
tf.app.flags.DEFINE_string("fac_wspec", "", "")
tf.app.flags.DEFINE_string("fac_img_dir", "", "")

# flags for trav
tf.app.flags.DEFINE_string("trav_spec", "", "")
tf.app.flags.DEFINE_string("trav_img_dir", "", "")

FLAGS = tf.app.flags.FLAGS

def main():
    if FLAGS.arch == "dnn":
        raise NotImplementedError()
    elif FLAGS.arch == "cnn":
        raise NotImplementedError()
    elif FLAGS.arch == "rnn":
        from models.rec_fhvae import RecFacHierVAE as model_class
        from parsers.model_parsers import rec_fhvae_model_parser as model_parser
    else:
        raise ValueError("unsupported architecture %s" % FLAGS.arch)
    
    # do training
    if not os.path.exists("%s/.done" % FLAGS.exp_dir):
        set_logger(custom_logger("%s/log/train.log" % FLAGS.exp_dir))
        is_train = True
        exp_dir, set_name, model_conf, train_conf, dataset_conf = _load_configs(
                FLAGS, model_parser, is_train)
        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        [train_set, dev_set, _], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                set_name, is_train, True, True, False)
        train(exp_dir, set_name, model, train_set, dev_set, 
                train_conf, FLAGS.n_print_steps, FLAGS.debug)
        unset_logger()
        
    # do testing
    if FLAGS.test:
        set_logger(custom_logger("%s/log/test.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf = _load_configs(
                FLAGS, model_parser, is_train)
        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                set_name, is_train, False, False, True)
        test(exp_dir, set_name, model, test_set)
        unset_logger()

    # do dumping latent variables
    if FLAGS.dump_latent:
        set_logger(custom_logger("%s/log/dump_latent.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf, \
                train_utt_id_map, dev_utt_id_map, test_utt_id_map, \
                dump_latent_opts, train_wspec, dev_wspec, test_wspec = \
                _load_dump_latent_configs(FLAGS, model_parser)

        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        info("WSPECS:\n\ttrain=%s, dev=%s, test=%s" % (train_wspec, dev_wspec, test_wspec))
        datasets, model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, set_name,
                is_train, bool(train_wspec), bool(dev_wspec), bool(test_wspec))
        wspecs = [train_wspec, dev_wspec, test_wspec]
        id_maps = [train_utt_id_map, dev_utt_id_map, test_utt_id_map]
        dataset_wspec_tuples = [t for t in zip(datasets, wspecs, id_maps) if bool(t[1])]
        for dataset, wspec, id_map in dataset_wspec_tuples:
            for path in wspec.split(":")[1].split(","):
                check_and_makedirs(os.path.dirname(path))
            writer = BFMWriter(wspec)
            dump_latent(exp_dir, model, dataset, id_map, writer.write, dump_latent_opts)
            writer.close()
        unset_logger()

    # do dumping representations
    if FLAGS.dump_repr:
        set_logger(custom_logger("%s/log/dump_repr.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf, \
                repr_set_name, train_repr_id_map, dev_repr_id_map, test_repr_id_map, \
                dump_repr_opts, train_wspec, dev_wspec, test_wspec = \
                _load_dump_repr_configs(FLAGS, model_parser)

        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        info("WSPECS:\n\ttrain=%s, dev=%s, test=%s" % (train_wspec, dev_wspec, test_wspec))
        datasets, model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, set_name,
                is_train, bool(train_wspec), bool(dev_wspec), bool(test_wspec))
        wspecs = [train_wspec, dev_wspec, test_wspec]
        id_maps = [train_repr_id_map, dev_repr_id_map, test_repr_id_map]
        dataset_wspec_tuples = [t for t in zip(datasets, wspecs, id_maps) if bool(t[1])]
        for dataset, wspec, id_map in dataset_wspec_tuples:
            for path in wspec.split(":")[1].split(","):
                check_and_makedirs(os.path.dirname(path))
            writer = BFVWriter(wspec)
            dump_repr(exp_dir, model, dataset, id_map, repr_set_name, writer.write, dump_repr_opts)
            writer.close()
        unset_logger()

    # do replacing utterance representation
    if FLAGS.repl_repr_utt:
        set_logger(custom_logger("%s/log/repl_repr_utt.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf, \
                repl_utt_set_name, repl_utt_wspec, repl_utt_id_map, \
                repl_utt_img_dir, repl_utt_list, label_str_to_repr = \
                _load_repl_repr_utt_configs(FLAGS, model_parser)

        info("Replacing Representation Configurations:")
        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        info("\tset_name: %s" % (repl_utt_set_name))
        info("\trepl_utt_list[0]: %s, repl_utt_wspec: %s" % (repl_utt_list[0], repl_utt_wspec))
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf,
                set_name, is_train, False, False, True)

        for path in repl_utt_wspec.split(":")[1].split(","):
            check_and_makedirs(os.path.dirname(path))
        with BFMWriter(repl_utt_wspec) as writer:
            write_fn = lambda uttid, feat_3d: writer.write(uttid, flatten_channel(feat_3d))
            repl_repr_utt(
                    exp_dir, model, test_set, repl_utt_set_name, 
                    repl_utt_id_map, label_str_to_repr, repl_utt_list, write_fn)

        check_and_makedirs(repl_utt_img_dir)
        plot_kaldi_feat(repl_utt_wspec, repl_utt_img_dir, dataset_conf["feat_cfg"]["feat_type"])
        unset_logger()

    # do visualizing factorization
    if FLAGS.fac:
        set_logger(custom_logger("%s/log/fac.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf, \
                fac_opts, fac_wspec, fac_img_dir = _load_fac_configs(FLAGS, model_parser)

        info("Factorization Configurations:")
        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        info("\tfac_wspec: %s" % fac_wspec)
        info("\tfac_opts: %s" % fac_opts)
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf,
                set_name, is_train, False, False, True)

        for path in fac_wspec.split(":")[1].split(","):
            check_and_makedirs(os.path.dirname(path))
        with BFMWriter(fac_wspec) as writer:
            write_fn = lambda uttid, feat_3d: writer.write(uttid, flatten_channel(feat_3d))
            factorize(exp_dir, model, test_set, write_fn, fac_opts)
        plot_kaldi_feat_fac(fac_wspec, fac_img_dir, dataset_conf["feat_cfg"]["feat_type"])
        unset_logger()

    # do traversing latent space
    if FLAGS.trav:
        set_logger(custom_logger("%s/log/trav.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, set_name, model_conf, train_conf, dataset_conf, \
                trav_img_dir, trav_opts = _load_trav_configs(FLAGS, model_parser)

        info("Traversing Configurations:")
        _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf)
        info("\ttrav_img_dir: %s" % trav_img_dir)
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf,
                set_name, is_train, False, False, True)

        traverse(exp_dir, model, test_set, trav_img_dir, trav_opts)
        unset_logger()

def _load_configs(flags, model_parser, is_train):
    """use fixed #batches for each epoch"""
    # load and copy configurations
    exp_dir = flags.exp_dir
    check_and_makedirs(exp_dir)
    set_name_path = "%s/set_name" % exp_dir
    
    if is_train:
        if os.path.exists("%s/model.cfg" % exp_dir):
            info("%s/model.cfg exists, using that" % exp_dir)
        else:
            model_conf = model_parser(flags.model_conf).get_config()
            if flags.n_latent1 > 0:
                model_conf["n_latent1"] = flags.n_latent1
            if flags.n_latent2 > 0:
                model_conf["n_latent2"] = flags.n_latent2
            with open("%s/model.cfg" % exp_dir, "w") as f:
                model_parser.write_config(model_conf, f)

        if os.path.exists("%s/train.cfg" % exp_dir):
            info("%s/train.cfg exists, using that" % exp_dir)
        else:
            train_conf = train_parser(flags.train_conf).get_config()
            if flags.alpha_dis != -1:
                train_conf["alpha_dis"] = flags.alpha_dis
            with open("%s/train.cfg" % exp_dir, "w") as f:
                train_parser.write_config(train_conf, f)

        maybe_copy(flags.dataset_conf, "%s/dataset.cfg" % exp_dir)
        
        if os.path.exists(set_name_path):
            info("%s exists, using that" % set_name_path)
        else:
            with open(set_name_path, "w") as f:
                f.write(flags.set_name)

    with open(set_name_path) as f:
        set_name = f.read().rstrip()
    model_conf = model_parser("%s/model.cfg" % exp_dir).get_config()
    train_conf = train_parser("%s/train.cfg" % exp_dir).get_config()
    dataset_conf = dataset_parser("%s/dataset.cfg" % exp_dir).get_config()
    if not is_train:
        train_conf["bs"] = 2048
    if not is_train and bool(flags.feat_rspec):
        dataset_conf["test_feat_rspec"] = flags.feat_rspec
        if bool(flags.feat_label_N) and bool(flags.feat_utt2label_path):
            dataset_conf["test_utt2label_paths"] = {}
            feat_set_name_list = flags.feat_set_name.split(",")
            label_N_list = flags.feat_label_N.split(",")
            utt2label_path_list = flags.feat_utt2label_path.split(",")
            assert(len(label_N_list) == len(feat_set_name_list))
            assert(len(label_N_list) == len(utt2label_path_list))
            for feat_set_name, N, path, in \
                    zip(feat_set_name_list, label_N_list, utt2label_path_list):
                dataset_conf["test_utt2label_paths"][feat_set_name] = (int(N), path)
        else:
            dataset_conf["test_utt2label_paths"] = {}
        dataset_conf["test_utt2talabels_paths"] = {}
        info("replaced test_feat_rspec with %s, utt2label_paths %s" % (
            dataset_conf["test_feat_rspec"], dataset_conf["test_utt2label_paths"]))

    return exp_dir, set_name, model_conf, train_conf, dataset_conf

def _print_configs(exp_dir, set_name, model_conf, train_conf, dataset_conf):
    if model_conf["n_bins"] != dataset_conf["n_bins"]:
        raise ValueError("model and dataset n_bins not matched (%s != %s)" % (
                model_conf["n_bins"], dataset_conf["n_bins"]))

    info("Experiment Directory:\n\t%s" % str(exp_dir))
    info("Set Name:\n\t%s" % str(set_name))
    info("Model Configurations:")
    for k, v in sorted(model_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    info("Training Configurations:")
    for k, v in sorted(train_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    info("Dataset Configurations:")
    for k, v in sorted(dataset_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    
def _load_datasets_and_model(
        FacHierVAE, dataset_conf, model_conf, train_conf,
        set_name, is_train, train, dev, test):
    # initialize dataset and model, create directories
    sets = datasets_loader(dataset_conf, train, dev, test)
    _set = [s for s in sets if s is not None][0]
    model_conf["n_class1"] = dataset_conf["train_utt2label_paths"][set_name][0]
    model_conf["input_shape"] = _set.feat_shape
    model_conf["target_shape"] = _set.feat_shape
    model_conf["target_dtype"] = tf.float32
    if model_conf["n_bins"] is not None:
        model_conf["target_dtype"] = tf.int32 
    model = FacHierVAE(model_conf, train_conf, training=is_train)
    return sets, model

def _load_dump_latent_configs(flags, model_parser):
    assert(flags.use_mean or flags.use_logvar)
    assert(flags.train_lat_wspec or flags.dev_lat_wspec or flags.test_lat_wspec)

    exp_dir, set_name, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    dataset_conf, utt_left_pad, utt_right_pad = \
            get_frame_ra_dataset_conf(dataset_conf)
    dump_latent_opts = {
            "use_z1": flags.use_z1,
            "use_z2": flags.use_z2,
            "use_mean": flags.use_mean, 
            "use_logvar": flags.use_logvar, 
            "utt_left_pad": utt_left_pad,
            "utt_right_pad": utt_right_pad}
    train_lat_wspec = flags.train_lat_wspec
    dev_lat_wspec = flags.dev_lat_wspec
    test_lat_wspec = flags.test_lat_wspec
    train_utt_id_map = _load_id_map(flags.train_utt_id_map) if train_lat_wspec else None
    dev_utt_id_map = _load_id_map(flags.dev_utt_id_map) if dev_lat_wspec else None
    test_utt_id_map = _load_id_map(flags.test_utt_id_map) if test_lat_wspec else None

    return exp_dir, set_name, model_conf, train_conf, dataset_conf, \
            train_utt_id_map, dev_utt_id_map, test_utt_id_map, \
            dump_latent_opts, train_lat_wspec, dev_lat_wspec, test_lat_wspec

def _load_dump_repr_configs(flags, model_parser):
    assert(flags.repr_set_name)
    assert(flags.which_repr in ["model_mu1", "mu1", "mu2"])
    assert(flags.train_repr_wspec or flags.dev_repr_wspec or flags.test_repr_wspec)

    exp_dir, set_name, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    # dataset_conf, _, _ = get_frame_ra_dataset_conf(dataset_conf)
    dataset_conf = get_nonoverlap_ra_dataset_conf(dataset_conf)
    dump_repr_opts = {"which_repr": flags.which_repr}
    train_repr_wspec = flags.train_repr_wspec
    dev_repr_wspec = flags.dev_repr_wspec
    test_repr_wspec = flags.test_repr_wspec
    train_repr_id_map = _load_id_map(flags.train_repr_id_map) if train_repr_wspec else None
    dev_repr_id_map = _load_id_map(flags.dev_repr_id_map) if dev_repr_wspec else None
    test_repr_id_map = _load_id_map(flags.test_repr_id_map) if test_repr_wspec else None
    repr_set_name = flags.repr_set_name

    return exp_dir, set_name, model_conf, train_conf, dataset_conf, \
            repr_set_name, train_repr_id_map, dev_repr_id_map, test_repr_id_map, \
            dump_repr_opts, train_repr_wspec, dev_repr_wspec, test_repr_wspec

def _load_repl_repr_utt_configs(flags, model_parser):
    assert(flags.repl_utt_set_name)
    assert(flags.repl_utt_repr_spec)
    assert(flags.repl_utt_wspec)
    assert(flags.repl_utt_list)
    assert(flags.repl_utt_img_dir)
    assert(flags.repl_utt_id_map)

    exp_dir, set_name, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    dataset_conf, _, _ = get_frame_ra_dataset_conf(dataset_conf)
    repl_utt_set_name = flags.repl_utt_set_name
    repl_utt_wspec = flags.repl_utt_wspec
    repl_utt_img_dir = flags.repl_utt_img_dir
    repl_utt_id_map = _load_id_map(flags.repl_utt_id_map)
    repl_utt_list = [line.rstrip().split() for line in open(flags.repl_utt_list)]
    label_str_to_repr = _load_repr(flags.repl_utt_repr_spec)

    return exp_dir, set_name, model_conf, train_conf, dataset_conf, \
            repl_utt_set_name, repl_utt_wspec, repl_utt_id_map, \
            repl_utt_img_dir, repl_utt_list, label_str_to_repr

def _load_fac_configs(flags, model_parser):
    assert(flags.fac_z1_spec)
    assert(flags.fac_z2_spec)
    assert(flags.fac_wspec)
    assert(flags.fac_img_dir)

    fac_n = flags.fac_n
    z1_utt_list, z1_segs = _load_utt_or_seg_spec(flags.fac_z1_spec)
    z2_utt_list, z2_segs = _load_utt_or_seg_spec(flags.fac_z2_spec)
    fac_opts = {
            "n": fac_n,
            "z1_utt_list": z1_utt_list,
            "z2_utt_list": z2_utt_list,
            "z1_segs": z1_segs,
            "z2_segs": z2_segs}
    fac_wspec = flags.fac_wspec
    fac_img_dir = flags.fac_img_dir
    exp_dir, set_name, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    return exp_dir, set_name, model_conf, train_conf, \
            dataset_conf, fac_opts, fac_wspec, fac_img_dir

def _load_trav_configs(flags, model_parser):
    assert(flags.trav_img_dir)

    exp_dir, set_name, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    if flags.trav_spec:
        seed_utt_list, seed_segs = _load_utt_or_seg_spec(flags.trav_spec)
    else:
        seed_utt_list, seed_segs = None, None
    trav_img_dir = flags.trav_img_dir
    trav_opts = {
        "n": 5,
        "seed_utt_list": seed_utt_list,
        "seed_segs": seed_segs,
        "k": 9,
        "trav_range": (-3, 3),
        "figsize": (10, 30),
        "feat_type": dataset_conf["feat_cfg"]["feat_type"],
    }
    return exp_dir, set_name, model_conf, train_conf, dataset_conf, trav_img_dir, trav_opts

def _load_utt_or_seg_spec(fac_spec_path):
    fac_spec = [line.rstrip() for line in open(fac_spec_path)]
    assert(bool(fac_spec))
    if len(fac_spec[0].split()) == 1:
        fac_list, fac_segs = fac_spec, None
    elif len(fac_spec[0].split()) == 3:
        fac_list = None
        fac_spec = [line.split() for line in fac_spec]
        fac_segs = [Segment(toks[0], int(toks[1]), int(toks[2]), None) for toks in fac_spec]
    else:
        raise ValueError(
                "#fields must be 1 (utterance id) or 3 (segment); got %s" % len(fac_spec[0]))
    return fac_list, fac_segs

def _load_id_map(id_map_path):
    """id_map_path is the path to a file with ``key id'' format
    output is a dictionary map from id (int) to key (str)"""
    id_map = {}
    with open(id_map_path) as f:
        id_content = f.readlines()
    for l in id_content:
        l = l.split(' ')
        id_map[int(l[1])] = l[0]
    return id_map

def _load_repr(repr_rspec):
    label_str_to_repr = dict()
    with SBFVReader(repr_rspec) as f:
        while not f.done():
            label_str, attr = f.next()
            label_str_to_repr[label_str] = attr
    return label_str_to_repr

if __name__ == "__main__":
    main()
