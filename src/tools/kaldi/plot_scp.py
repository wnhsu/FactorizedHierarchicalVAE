from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from utils import *
from tools.vis import plot_rows
from tools.kaldi.prep_kaldi_feat import unflatten_channel
from kaldi_io import SequentialBaseFloatMatrixReader as SBFMReader

def plot_kaldi_feat(wspec, img_dir, feat_type="fbank_raw"):
    print("plotting wspec %s, save images to %s" % (wspec, repr(img_dir)))
    mode = "show" if img_dir is None else "save"
    if img_dir is not None:
        check_and_makedirs(img_dir)
    img_h = 4.
    with SBFMReader(wspec) as f:
        while not f.done():
            utt_id, utt_feats = f.next()
            img_w = img_h / utt_feats.shape[1] * utt_feats.shape[0]
            utt_feats = unflatten_channel(utt_feats, 1)[np.newaxis, ...]
            plot_rows(
                    [utt_feats], utt_id, feat_type=feat_type, 
                    mode=mode, name=os.path.join(img_dir, "%s.png" % utt_id),
                    figsize=(img_w, img_h))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("wspec")
    parser.add_argument("img_dir")
    parser.add_argument("--feat_type", default="spec")
    
    args = parser.parse_args()
    plot_kaldi_feat(args.wspec, args.img_dir, args.feat_type)
