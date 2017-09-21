from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from utils import *
from tools.vis import plot_grids
from tools.kaldi.prep_kaldi_feat import unflatten_channel
from kaldi_io import SequentialBaseFloatMatrixReader as SBFMReader

def plot_kaldi_feat_fac(rspec, img_dir, feat_type="fbank_raw"):
    """
    visualizing factorization utt_id of format: 
    "%s_%s_%s_%s" % (i, j, utt_id_i, utt_id_j), 
    for i in range(n_i), j in range(n_j)
    """
    print("plot factorization:")
    print("\tfeat_rspec %s, save images to %s" % (rspec, repr(img_dir)))
    mode = "show" if img_dir is None else "save"
    if img_dir is not None:
        check_and_makedirs(img_dir)
    toks_feats_list = [
            (tup[0].split("_"), unflatten_channel(tup[1], 1)[np.newaxis, ...]) \
            for tup in SBFMReader(rspec)]
    
    fac_utt_id_toks_list, fac_feats_list = zip(
            *[(tup[0], tup[1]) for tup in toks_feats_list
            if int(tup[0][0]) > -1 and int(tup[0][1]) > -1])
    n_i = max([int(toks[0]) for toks in fac_utt_id_toks_list]) + 1
    n_j = max([int(toks[1]) for toks in fac_utt_id_toks_list]) + 1
    feats_shape = fac_feats_list[0].shape
    fac_feats_list = np.asarray(fac_feats_list).reshape([n_i, n_j] + list(feats_shape))
    figsize = (3.5 * n_i, 5. * n_j)
    img_path = "%s/fac.png" % img_dir
    plot_grids(fac_feats_list, feat_type=feat_type, mode=mode, name=img_path, figsize=figsize)

    X1_utt_id_toks_list, X1_feats_list = zip(
            *[(tup[0], tup[1]) for tup in toks_feats_list if int(tup[0][1]) == -1])
    X1_feats_list = np.asarray(X1_feats_list).reshape([n_i, 2] + list(feats_shape))
    figsize = (3.5 * n_i, 5. * 2)
    img_path = "%s/X1.png" % img_dir
    plot_grids(X1_feats_list, feat_type=feat_type, mode=mode, name=img_path, figsize=figsize)
    
    X2_utt_id_toks_list, X2_feats_list = zip(
            *[(tup[0], tup[1]) for tup in toks_feats_list if int(tup[0][0]) == -1])
    X2_feats_list = np.asarray(X2_feats_list).reshape([n_j, 2] + list(feats_shape))
    X2_feats_list = X2_feats_list.transpose((1, 0, 2, 3, 4, 5))
    figsize = (3.5 * 2, 5. * n_j)
    img_path = "%s/X2.png" % img_dir
    plot_grids(X2_feats_list, feat_type=feat_type, mode=mode, name=img_path, figsize=figsize)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("rspec")
    parser.add_argument("img_dir")
    parser.add_argument("--feat_type", default="spec")
    
    args = parser.parse_args()
    plot_kaldi_feat_fac(args.rspec, args.img_dir, args.feat_type)
