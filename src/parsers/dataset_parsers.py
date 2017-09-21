#!/usr/bin/python

from collections import OrderedDict
from parser_common import *

def parse_label_paths(raw_str):
    """
    raw str in the format of: 
        [name_1:]n_class_1:path_1,[name_2:]n_class_2:path_2,...
    """
    raw_toks = [tok.split(":") for tok in raw_str.split(",") if tok]
    if raw_toks and len(raw_toks[0]) == 3:
        raw_toks = OrderedDict([(tok[0], (int(tok[1]), tok[2])) for tok in raw_toks])
    return raw_toks

class base_dataset_parser(base_parser):
    def __init__(self, dataset_config_path):
        self.parser = DefaultConfigParser()
        
        parser = self.parser
        config = {}
        if len(self.parser.read(dataset_config_path)) == 0:
            raise ValueError("dataset_parser(): %s not found", dataset_config_path)
 
        config["fmt"]           = parser.get("data", "fmt")
        config["egs"]           = parser.get("data", "egs")
        config["mvn_path"]      = parser.get("data", "mvn_path", None, False)
        config["use_chan"]      = [int(chan) for chan in parser.get(
                                  "data", "use_chan", "0").split(',')]
        config["remove_0th"]    = parser.getboolean("data", "remove_0th", False)
        config["max_to_load"]   = parser.getint("data", "max_to_load", -1)
        config["if_rand"]       = parser.getboolean("data", "if_rand", True)

        feat_cfg = {}
        feat_cfg["feat_type"]   = parser.get("data", "feat_type", None, False)
        feat_cfg["decom"]       = parser.get("data", "decom", None, False)
        feat_cfg["add_dc"]      = config["remove_0th"]

        if parser.has_section("stft"):
            stft_cfg = {}
            stft_cfg["fs"]               = parser.getint("stft", "fs", 16000)
            stft_cfg["frame_size_n"]     = parser.getint("stft", "frame_size_n", 400)
            stft_cfg["shift_size_n"]     = parser.getint("stft", "shift_size_n", 160)
            stft_cfg["fft_size"]         = parser.getint("stft", "fft_size", 400) 
        else:
            stft_cfg = None

        feat_cfg["stft_cfg"] = stft_cfg
        config["feat_cfg"] = feat_cfg
        self.config = config

class kaldi_ra_dataset_parser(base_dataset_parser):
    def __init__(self, dataset_config_path):
        super(kaldi_ra_dataset_parser, self).__init__(dataset_config_path)

        parser = self.parser
        config = {}

        config["n_chan"]                = parser.getint("data", "n_chan")
        config["seg_len"]               = parser.getint("data", "seg_len")
        config["seg_shift"]             = parser.getint("data", "seg_shift")
        config["seg_rand"]              = parser.getboolean("data", "seg_rand")
        config["train_feat_rspec"]      = parser.get("data", "train_feat_rspec", None, False)
        config["dev_feat_rspec"]        = parser.get("data", "dev_feat_rspec", None, False)
        config["test_feat_rspec"]       = parser.get("data", "test_feat_rspec", None, False)     
        config["train_utt2label_paths"] = parse_label_paths(parser.get("data", "train_utt2label_paths", ""))
        config["dev_utt2label_paths"]   = parse_label_paths(parser.get("data", "dev_utt2label_paths", ""))
        config["test_utt2label_paths"]  = parse_label_paths(parser.get("data", "test_utt2label_paths", ""))
        config["train_utt2talabels_paths"] = parse_label_paths(parser.get("data", "train_utt2talabels_paths", ""))
        config["dev_utt2talabels_paths"]   = parse_label_paths(parser.get("data", "dev_utt2talabels_paths", ""))
        config["test_utt2talabels_paths"]  = parse_label_paths(parser.get("data", "test_utt2talabels_paths", ""))
        
        config["n_bins"]                 = parser.getint("data", "n_bins", None, False)
        lim_raw_str = parser.get("data", "lim", "")
        config["lim"]                    = [float(i) for i in lim_raw_str.split(',') if i]
        config["q_type"]                 = parser.get("data", "q_type", "mu", False)

        for k in config:
            self.config[k] = config[k]

    def get_config(self):
        return self.config
