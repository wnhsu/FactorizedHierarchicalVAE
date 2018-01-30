#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

# prepare dataset config for VAE/FHVAE training 
# assume having kaldi/egs/timit/s5/data ready

stage=0
nj=40

TIMIT_RAW_DATA=
TIMIT_KALDI_EGS=
KALDI_ROOT=

egs_dir=$(pwd)/data
fbank_conf=$(pwd)/conf/fbank.conf

tr=train 
dt=dev
tt=test

seg_len=20
seg_shift=20
seg_rand=True

dataset_name=dataset.cfg

. ./path.sh 
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

set -eu

required="$tr $dt $tt"
map_file=$TIMIT_KALDI_EGS/conf/phones.60-48-39.map
dataset_cfg=$egs_dir/fbank_scp/$tr/$dataset_name

if [ $stage -le 0 ]; then 
    echo "$0: stage 0, process raw data"
    local/timit_raw_data_prep.sh \
        $TIMIT_RAW_DATA $KALDI_ROOT $TIMIT_KALDI_EGS || exit 1;
fi

if [ $stage -le 1 ]; then
    for d in $required; do
        steps/make_fbank.sh --nj $nj --fbank_conf $fbank_conf \
            $TIMIT_KALDI_EGS data/$d data/${d}_fbank $egs_dir/fbank_scp/$d || exit 1;
    done
fi

if [ $stage -le 2 ]; then
    echo "$0: stage 2, prepare utt2uttid and utt2spkid"
    for d in $egs_dir/fbank_scp/{$tr,$dt,$tt}; do
        steps/make_utt2labels.sh --isutt true \
            $d/feats.scp $d/utt2uttid || exit 1;
        steps/make_utt2labels.sh --isutt false \
            $d/utt2spk $d/utt2spkid $d/spk2spkid || exit 1;
    done
fi

if [ $stage -le 3 ]; then
    echo "$0: stage 3, prepare utt2phoneid time-aligned labels"
    for s in $tr $dt $tt; do
        data_dir=$egs_dir/fbank_scp/$s
        phn_scp=$TIMIT_KALDI_EGS/data/local/data/${d}_phn.scp
        python src/tools/kaldi/phn_to_talabel.py \
            $phn_scp $map_file $data_dir/utt2phoneid.talabel \
            $data_dir/phone2phoneid || exit 1;
    done
fi

if [ $stage -le 4 ]; then
    echo "$0: stage 4, generate dataset.cfg"
    steps/make_dataset_conf.sh --hasspk true --hasphone true \
        --egs timit --feat_type fbank_raw \
        --seg_len $seg_len --seg_shift $seg_shift --seg_rand $seg_rand \
        $egs_dir/fbank_scp/$tr $egs_dir/fbank_scp/$dt $egs_dir/fbank_scp/$tt \
        $dataset_cfg || exit 1;
fi
