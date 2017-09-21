"""General Datasets Loaders"""
import os
from . import *
from .kaldi_ra_dataset import KaldiRADataset

def get_frame_ra_dataset_conf(dataset_conf):
    """extract segments by shifting one frame, 
    return left/right padding needed"""

    if dataset_conf["if_rand"]:
        info("disabled dataset_conf if_rand")
        dataset_conf["if_rand"] = False
    if dataset_conf["seg_rand"]:
        info("disabled dataset_conf seg_rand")
        dataset_conf["seg_rand"] = False
    if dataset_conf["seg_shift"] != 1:
        info("change seg_shift from %s to 1" % dataset_conf["seg_shift"])
        dataset_conf["seg_shift"] = 1
    utt_left_pad = int(np.floor((dataset_conf["seg_len"] - 1) / 2.))
    utt_right_pad = int(np.ceil((dataset_conf["seg_len"] - 1) / 2.))
    info("pad %s frames at the beginning of each utterance" % utt_left_pad)
    info("pad %s frames at the end of each utterance" % utt_right_pad)
    return dataset_conf, utt_left_pad, utt_right_pad

def get_nonoverlap_ra_dataset_conf(dataset_conf):
    """extract segments by shifting segment length"""
    if dataset_conf["if_rand"]:
        info("disabled dataset_conf if_rand")
        dataset_conf["if_rand"] = False
    if dataset_conf["seg_rand"]:
        info("disabled dataset_conf seg_rand")
        dataset_conf["seg_rand"] = False
    if dataset_conf["seg_shift"] != dataset_conf["seg_len"]:
        info("change seg_shift from %s to %s" % (
            dataset_conf["seg_shift"], dataset_conf["seg_len"]))
        dataset_conf["seg_shift"] = dataset_conf["seg_len"]
    return dataset_conf

def datasets_loader(conf, train=True, dev=True, test=True):
    """
    simple wrapper for unified interface of datasets loaders
    """
    fmt = conf.pop("fmt")
    if fmt == "kaldi_ra":
        return kaldi_ra_datasets_loader(conf, train, dev, test)
    else:
        raise ValueError("dataset format %s not supported" % fmt)

def kaldi_ra_datasets_loader(conf, train=True, dev=True, test=True):
    # not quantizing if n_bins is None
    if conf["n_bins"] is None:
        Dataset = KaldiRADataset
    else:
        raise NotImplementedError
    return _kaldi_ra_datasets_loader(Dataset, conf, train, dev, test)

def _kaldi_ra_datasets_loader(Dataset, conf, train=True, dev=True, test=True):
    """
    loading train/dev/test sets according to conf.
        
        - `if_rand' is disabled for dev/test. 
        - `seg_rand' is disabled for dev/test. 
        - if `mvn_path' is specified, `mvn_path' need to already exist 
          or `train' is set to True so `train' would compute it
        
    """
    info("Loading datasets: train=%s, dev=%s, test=%s" % (train, dev, test))
    train_feat_rspec            = conf["train_feat_rspec"]
    dev_feat_rspec              = conf["dev_feat_rspec"]
    test_feat_rspec             = conf["test_feat_rspec"]
    train_utt2label_paths       = conf["train_utt2label_paths"]
    dev_utt2label_paths         = conf["dev_utt2label_paths"]
    test_utt2label_paths        = conf["test_utt2label_paths"]
    train_utt2talabels_paths    = conf["train_utt2talabels_paths"]
    dev_utt2talabels_paths      = conf["dev_utt2talabels_paths"]
    test_utt2talabels_paths     = conf["test_utt2talabels_paths"]
    seg_len                     = conf["seg_len"]
    seg_shift                   = conf["seg_shift"]
    seg_rand                    = conf["seg_rand"]
    n_chan                      = conf["n_chan"]
    use_chan                    = conf["use_chan"]
    use_fbin                    = slice(1, None) if conf["remove_0th"] else slice(None, None)
    if_rand                     = conf["if_rand"]
    mvn_path                    = conf["mvn_path"]      
    max_to_load                 = conf["max_to_load"]

    n_bins                      = conf["n_bins"]
    lim                         = conf["lim"]
    q_type                      = conf["q_type"]

    apply_mvn       = bool(mvn_path)

    assert(not train or train_feat_rspec)
    assert(not dev or dev_feat_rspec)
    assert(not test or test_feat_rspec)
    assert(not apply_mvn or (os.path.exists(mvn_path) or train))

    train_dataset   = None
    dev_dataset     = None
    test_dataset    = None
    
    if train:
        train_dataset = Dataset(feat_rspec=train_feat_rspec, 
                                seg_len=seg_len,
                                seg_shift=seg_shift,
                                seg_rand=seg_rand,
                                n_chan=n_chan, 
                                use_chan=use_chan, 
                                use_fbin=use_fbin,
                                if_rand=if_rand,
                                mvn_path=mvn_path,
                                max_to_load=max_to_load,
                                utt2label_paths=train_utt2label_paths,
                                utt2talabels_paths=train_utt2talabels_paths,
                                n_bins=n_bins,
                                lim=lim,
                                q_type=q_type)
    if dev:
        dev_dataset = Dataset(feat_rspec=dev_feat_rspec, 
                              seg_len=seg_len,
                              seg_shift=seg_shift,
                              seg_rand=False,
                              n_chan=n_chan, 
                              use_chan=use_chan, 
                              use_fbin=use_fbin,
                              if_rand=False,
                              mvn_path=mvn_path,
                              max_to_load=max_to_load,
                              utt2label_paths=dev_utt2label_paths,
                              utt2talabels_paths=dev_utt2talabels_paths,
                              n_bins=n_bins,
                              lim=lim,
                              q_type=q_type)
    if test:
        test_dataset = Dataset(feat_rspec=test_feat_rspec, 
                               seg_len=seg_len,
                               seg_shift=seg_shift,
                               seg_rand=False,
                               n_chan=n_chan, 
                               use_chan=use_chan, 
                               use_fbin=use_fbin,
                               if_rand=False,
                               mvn_path=mvn_path,
                               max_to_load=max_to_load,
                               utt2label_paths=test_utt2label_paths,
                               utt2talabels_paths=test_utt2talabels_paths,
                               n_bins=n_bins,
                               lim=lim,
                               q_type=q_type)
    
    return train_dataset, dev_dataset, test_dataset
