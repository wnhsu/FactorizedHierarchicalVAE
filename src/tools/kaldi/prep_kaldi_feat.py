import sys
import numpy as np
import re 
import time
import soundfile as sf
from collections import defaultdict

from utils import *
from tools.audio import comp_spec_image
from kaldi_io import BaseFloatMatrixWriter as BFMWriter

def flatten_channel(utt_feats):
    """
    convert a 3D tensor of (C, T, F) to (T, F_c1+F_c2+...)
    """
    assert(isinstance(utt_feats, np.ndarray) and utt_feats.ndim==3)
    return np.concatenate(utt_feats, axis=1)

def unflatten_channel(utt_feats, n_chan):
    """
    convert a 2D flattened tensor of (T, F_c1+F_c2+...) to (C, T, F)
    """
    assert(isinstance(utt_feats, np.ndarray) and utt_feats.ndim==2)
    utt_feats = utt_feats.reshape((utt_feats.shape[0], n_chan, -1))
    utt_feats = utt_feats.transpose((1, 0, 2))
    return utt_feats

def dump_kaldi_spec(wav_scp, out_basename, feat_opts):
    if os.path.exists("%s.scp" % out_basename):
        info("%s.scp exists. skipped..." % out_basename)
        return
    
    check_and_makedirs(os.path.dirname(out_basename))
    with open(wav_scp) as f:
        wav_paths = [line.rstrip().split() for line in f]
    
    out_basename = os.path.abspath(out_basename)
    wspec = "ark,scp:%s.ark,%s.scp" % (out_basename, out_basename)
    start_time = time.time()
    with BFMWriter(wspec) as f:
        for i_u, (utt_id, wav_path) in enumerate(wav_paths):
            wav, fs = sf.read(wav_path)
            if wav.ndim > 1:
                info("%s has more than one channels. " % wav_path + \
                        "only first one will be used")
                wav = wav[:,0]
            spec_image = comp_spec_image(wav, **feat_opts)
            # flatten over channels: (2, T, F) -> (T, F_c1+F_c2)
            feat = np.concatenate([spec_image[0], spec_image[1]], axis=1)
            f.write(utt_id, feat)
            if (i_u + 1) % 100 == 0:
                info("processed %s utterances" % (i_u + 1))
    info("time elapsed: %.2f seconds" % (time.time() - start_time))
