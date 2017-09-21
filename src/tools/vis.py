import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from tools import *
from tools.audio import MelBank

FBANK_RANGE = (-8., 8.)
FBANK_RAW_RANGE = (0., 20.)
SPEC_RANGE = (-76., 0.)
IM_RANGE = (0., 1.)
RAW_RANGE = (-.1, .1)

def _normalize_image(image, d_range):
    image = image.clip(*d_range)
    image = (image - d_range[0]) / (d_range[1] - d_range[0])
    return image

def _add_sep(images):
    assert(images.ndim == 4)

    sep = np.ones(list(images.shape[1:-1])+[1]) * np.max(images)
    image_concat = images[0]
    for image in images[1:]:
        image_concat = np.concatenate([image_concat, sep, image], axis=-1)
    image_concat = image_concat.reshape(image_concat.shape[1:])
    return image_concat
    
def adjust_ticklabels(ax, egs="timit", feat_type=DEFAULT_FEAT_TYPE, 
                      rm_xticklabels=True, n_intvl=4):
    ax.set_frame_on(False)
    if is_audio(egs):
        if feat_type.startswith("fbank"):
            mel_banks = MelBank(low_freq=20, high_freq=8000, num_bins=80, 
                                sample_freq=16000, frame_size=32)
            y_labels = mel_banks.center_freqs
        elif feat_type == "spec":
            y_labels = np.linspace(0, 8000, 201)
        else:
            raise ValueError("feat type %s invalid" % feat_type)
        
        y_labels = ["%.f" % f for f in y_labels]
        ytick_index = map(int, map(round, 
            np.linspace(0, len(y_labels), n_intvl+1)[1:] - 1))

        ax.set_yticks(ytick_index, minor=False)
        ax.set_yticklabels(np.asarray(y_labels)[ytick_index], minor=False)
        ax.invert_yaxis()

    if rm_xticklabels:
        ax.xaxis.set_ticks_position("none")
        ax.set_xticklabels([''])

def adjust_clim(im, egs="timit", feat_type=DEFAULT_FEAT_TYPE):
    if is_audio(egs):
        if feat_type == "fbank":
            im.set_clim(FBANK_RANGE)
        elif feat_type == "fbank_raw":
            im.set_clim(FBANK_RAW_RANGE)
        elif feat_type == "spec":
            im.set_clim(SPEC_RANGE)
        else:
            raise ValueError("feat type %s invalid" % feat_type)
    else:
        im.set_clim(IM_RANGE)

def im_post_proc(images, egs="timit"):
    if is_audio(egs):
        axes = range(np.asarray(images.ndim))
        axes[-2:] = axes[-2:][::-1]
        images = images.transpose(axes)
    else:
        images[np.where(images<0)] = 0
        images[np.where(images>1)] = 1
    return images

def imshow(image, ax, egs="timit", feat_type=None):
    if is_audio(egs):
        im = ax.imshow(image, cmap="Greys", interpolation="none", aspect="auto")
    else:
        im = ax.imshow(image, cmap="Greys_r", interpolation="none", aspect="auto")
    adjust_ticklabels(ax, egs, feat_type)
    adjust_clim(im, egs, feat_type)
    
def plot_rows(feats_list, labels=None, egs="timit", 
              feat_type=DEFAULT_FEAT_TYPE, mode="show", 
              name=None, figsize=(16, 12)):
    """
    feats_list     : a list of 4-D tensors (n, c, h, w)
    """
    fig = plt.figure(figsize=figsize)
    n_rows = len(feats_list)
    n_ch = feats_list[0].shape[1]

    if isinstance(labels, str):
        labels = [labels] + [""] * (n_rows - 1)
    elif labels is None:
        labels = [""] * n_rows
    else:
        assert(len(labels) == n_rows)

    if feat_type == "raw":
        assert(n_ch == 1)
        n_cols = len(feats_list[0])

    for i in xrange(n_rows):
        images = feats_list[i]
        images = im_post_proc(images, egs)
        if feat_type == "raw":
            # raw waveform
            for j, image in enumerate(images):
                ax = fig.add_subplot(n_rows, n_cols, i*n_cols+j+1)
                im = ax.plot(image.reshape((-1,)))
                ax.set_ylim(RAW_RANGE)
                # ax.set_ylabel("amplitude")
                # ax.set_title("%s" % (labels[i]))
        else:
            # image
            for ch in xrange(n_ch):
                image = _add_sep(images[:, ch:ch+1, :, :])
                ax = fig.add_subplot(n_rows*n_ch, 1, i*n_ch+ch+1)
                if ch == 0:
                    ax.set_ylabel("freq(Hz)")
                    ax.set_title("%s" % (labels[i]))
                imshow(image, ax, egs, feat_type)
    ax.set_xlabel("time")
    
    if mode == "show":
        plt.show()
    elif mode == "save":
        assert(not name is None)
        plt.savefig(name, bbox_inches='tight')

    plt.close(fig)

def plot_grids(images_mat, labels=None, egs="timit", 
               feat_type=DEFAULT_FEAT_TYPE, mode="show", 
               name=None, figsize=(16, 12)):
    """
    images_mat      : images_mat[j][i] is 4-D tensors (n, c, h, w) for row i, col j
    """
    fig = plt.figure(figsize=figsize)
    n_cols = len(images_mat)
    n_rows = len(images_mat[0])
    n_ch = images_mat[0][0].shape[1]
    # log("%s rows, %s cols, %s ch" % (n_rows, n_cols, n_ch))

    if not labels is None:
        assert(len(labels) == n_cols)
    else:
        labels = [[" "] * n_rows] * n_cols

    width_ratios = [l[0].shape[0] * l[0].shape[3] for l in images_mat]
    gs = gridspec.GridSpec(n_rows*n_ch, n_cols, width_ratios=width_ratios)
    for j in xrange(n_cols):
        feats_list = images_mat[j]
        for i in xrange(n_rows):
            images = feats_list[i]
            images = im_post_proc(images, egs)
            for ch in xrange(n_ch):
                image = _add_sep(images[:, ch:ch+1, :, :])
                # ax = fig.add_subplot(n_rows*n_ch, n_cols, (i*n_ch+ch)*n_cols+j+1)
                ax = plt.subplot(gs[(i*n_ch+ch)*n_cols+j])
                if ch == 0 and j == 0:
                    pass
                    # ax.set_ylabel("freq(Hz)")
                    # ax.set_title("%s" % (labels[j][i]))
                imshow(image, ax, egs, feat_type)
        # ax.set_xlabel("time")

    if mode == "show":
        plt.show()
    elif mode == "save":
        assert(not name is None)
        plt.savefig(name, bbox_inches='tight')

    plt.close(fig)

def plot_heatmap(X, labels_row, labels_col, mode="show", name=None, figsize=(8, 11)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    heatmap = ax.pcolor(X, cmap=plt.cm.Blues, alpha=0.8)
    cbar = plt.colorbar(heatmap)

    # turn off the frame
    ax.set_frame_on(False)
    
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    # Set the labels
    ax.set_xticklabels(labels_col, minor=False)
    ax.set_yticklabels(labels_row, minor=False)
    
    # rotate the
    plt.xticks(rotation=45)
    
    ax.grid(False)
    
    # Turn off all the ticks
    ax = plt.gca()
    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

    if mode == "show":
        plt.show()
    elif mode == "save":
        assert(not name is None)
        plt.savefig(name, bbox_inches='tight')

    plt.close(fig)
