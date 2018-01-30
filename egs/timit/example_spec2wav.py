#!/usr/bin/python

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

import numpy as np
from parsers.dataset_parsers import kaldi_ra_dataset_parser
from datasets.datasets_loaders import datasets_loader
from tools.audio import convert_to_complex_spec, complex_spec_to_audio

# load dataset and dataset configuration
d_conf = kaldi_ra_dataset_parser("data/spec_scp/train/dataset.cfg").get_config()
feat_cfg = d_conf["feat_cfg"]
print "\nSTFT configuration:"
print "\n".join([str(k).ljust(15) + str(v) for k, v in feat_cfg.iteritems()]) + "\n"
[_, _, tt_dset] = datasets_loader(d_conf, False, False, True)

# collect utterance log magnitude spectrogram
utt, idx = "fdhc0_si1559", 1
utt_feats = []
for feats, _, _, _ in tt_dset.iterator_by_label(2048, "uttid", idx):
    utt_feats.append(feats)
utt_feats = np.concatenate(utt_feats, axis=0)
logmagspec = np.concatenate(tt_dset.undo_mvn(utt_feats), axis=1)
assert(logmagspec.shape[0] == 1)

# estimate phase spectrogram from log magnitude spectrogram
est_phase_opts = {
        "frame_size_n": feat_cfg["stft_cfg"]["frame_size_n"],
        "shift_size_n": feat_cfg["stft_cfg"]["shift_size_n"],
        "fft_size": feat_cfg["stft_cfg"]["fft_size"]}
complex_spec = convert_to_complex_spec(
        logmagspec, None, feat_cfg["decom"], "est", feat_cfg["add_dc"], est_phase_opts)

# write reconstructed waveform
out_path = "%s_griffinlim.wav" % utt
complex_spec_to_audio(complex_spec, out_path, trim=20, **feat_cfg["stft_cfg"])
