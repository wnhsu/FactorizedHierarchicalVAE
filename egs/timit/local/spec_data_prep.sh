#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

# prepare dataset config for VAE/FHVAE training

stage=0
nj=40

TIMIT_RAW_DATA=
TIMIT_KALDI_EGS=
KALDI_ROOT=

egs_dir=$(pwd)/data

required="train dev test"
vae_tr="train"
vae_dt="dev"
vae_tt="test"

fs=16000
fn=400
sn=160
fft=400

seg_len=20
seg_shift=20
seg_rand=True

dataset_name=dataset.cfg

. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

set -eu

map_file=$TIMIT_KALDI_EGS/conf/phones.60-48-39.map
dataset_cfg=$egs_dir/spec_scp/${vae_tr}/$dataset_name

if [ $stage -le 0 ]; then 
    echo "$0: stage 0, process raw data"
    local/timit_raw_data_prep.sh \
        $TIMIT_RAW_DATA $KALDI_ROOT $TIMIT_KALDI_EGS || exit 1;
fi

if [ $stage -le 1 ]; then 
    echo "$0: stage 1, check and make spec features"
    for d in $required; do
        steps/make_spec.sh $TIMIT_KALDI_EGS/data/$d \
            $egs_dir/wav/$d $egs_dir/spec_scp/$d || exit 1;
    done
fi

if [ $stage -le 2 ]; then
    echo "$0: stage 2, prepare utt2uttid and utt2spkid"
    for d in $required; do
        steps/make_utt2labels.sh --isutt true \
            $egs_dir/spec_scp/$d/feats.scp \
            $egs_dir/spec_scp/$d/utt2uttid || exit 1;
        steps/make_utt2labels.sh --isutt false \
            $egs_dir/spec_scp/$d/utt2spk $egs_dir/spec_scp/$d/utt2spkid \
            $egs_dir/spec_scp/$d/spk2spkid || exit 1;
    done
fi

if [ $stage -le 3 ]; then
    echo "$0: stage 3, prepare utt2phoneid time-aligned labels"
    for d in $required; do
        data_dir=$egs_dir/spec_scp/$d
        phn_scp=$TIMIT_KALDI_EGS/data/local/data/${d}_phn.scp
        python src/tools/kaldi/phn_to_talabel.py \
            $phn_scp $map_file $data_dir/utt2phoneid.talabel \
            $data_dir/phone2phoneid || exit 1;
    done
fi

if [ $stage -le 4 ]; then
    echo "$0: stage 4, generate dataset.cfg"
    steps/make_dataset_conf.sh --hasspk true --hasphone true \
        --hasstft true --egs timit \
        --fs $fs --fn $fn --sn $sn --fft $fft --feat_type spec \
        --n_chan 2 --use_chan 0 --remove_0th True --decom mp \
        --seg_len $seg_len --seg_shift $seg_shift --seg_rand $seg_rand \
        $egs_dir/spec_scp/${vae_tr} $egs_dir/spec_scp/${vae_dt} \
        $egs_dir/spec_scp/${vae_tt} $dataset_cfg || exit 1;
fi
