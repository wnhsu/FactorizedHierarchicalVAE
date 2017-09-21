#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

TIMIT_DIR=../../kaldi/egs/timit/s5
TIMIT_RAW_DATA=/usr/users/dharwath/data/timit

feat_type=spec

# input sections
exp_dir=exp/fhvae_lstm1L256_lat32_32_ad10_${feat_type}_20_20_uttid
train_conf=conf/train/fhvae/e500_p50_lr1e-3_bs256_nbs2000_ad10.cfg
model_conf=conf/model/fhvae/lstm_1L_256_lat_32_32.cfg
dataset_conf=data/${feat_type}_scp/train/dataset.cfg
set_name=uttid

feat_dir=data/${feat_type}_scp
tr=train
dt=dev
tt=test

fac_label=qual_conf/fac.txt
tr_repl_utt_label=qual_conf/repl_utt.txt

stage=0

. ./path.sh 
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

set -eu

if [ $stage -le -1 ]; then
    ./local/${feat_type}_data_prep.sh --TIMIT_RAW_DATA $TIMIT_RAW_DATA \
        --TIMIT_KALDI_EGS $TIMIT_DIR --KALDI_ROOT $KALDI_DIR || exit 1;
fi

tr_feat_rspec=scp:$feat_dir/$tr/feats.scp
tr_utt2uttid=$feat_dir/$tr/utt2uttid

dt_feat_rspec=scp:$feat_dir/$dt/feats.scp
dt_utt2uttid=$feat_dir/$dt/utt2uttid

tt_feat_rspec=scp:$feat_dir/$tt/feats.scp
tt_utt2uttid=$feat_dir/$tt/utt2uttid

# output sections
tr_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/train.ark
dt_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/dev.ark
tt_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/test.ark
tr_repr_wspec=ark:$exp_dir/eval/repr/train.ark
dt_repr_wspec=ark:$exp_dir/eval/repr/dev.ark
tt_repr_wspec=ark:$exp_dir/eval/repr/test.ark

tr_repl_utt_wspec=ark:$exp_dir/eval/repl_utt/train.ark
tr_repl_utt_img_dir=$exp_dir/eval/repl_utt/img

fac_wspec=ark:$exp_dir/eval/fac/test.ark
fac_img_dir=$exp_dir/eval/fac/img

trav_img_dir=$exp_dir/eval/traverse

if [ $stage -le 0 ]; then
    echo "$0: stage 0, start FHVAE training ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --set_name=$set_name --dataset_conf=$dataset_conf \
        --train_conf=$train_conf --model_conf=$model_conf || exit 1;

    echo "$0: finished FHVAE training ($(date))"
fi

if [ $stage -le 1 ]; then
    echo "$0: stage 1, visualize disentanglement ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest --fac \
        --fac_z1_spec=$fac_label --fac_z2_spec=$fac_label \
        --fac_wspec=$fac_wspec --fac_img_dir=$fac_img_dir || exit 1;
fi

if [ $stage -le 2 ]; then
    echo "$0: stage 2, dump latent variables and s-vectors ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --dump_lat --use_mean --use_logvar --dump_z2 \
        --train_lat_wspec=$tr_z2_meanvar_wspec --train_utt_id_map=$tr_utt2uttid \
        --dev_lat_wspec=$dt_z2_meanvar_wspec --dev_utt_id_map=$dt_utt2uttid \
        --test_lat_wspec=$tt_z2_meanvar_wspec --test_utt_id_map=$tt_utt2uttid \
        --dump_repr --repr_set_name=uttid \
        --train_repr_wspec=$tr_repr_wspec --train_repr_id_map=$tr_utt2uttid \
        --dev_repr_wspec=$dt_repr_wspec --dev_repr_id_map=$dt_utt2uttid \
        --test_repr_wspec=$tt_repr_wspec --test_repr_id_map=$tt_utt2uttid || exit 1;
fi

if [ $stage -le 3 ]; then
    echo "$0: stage 3, replacing utterance s-vector ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --feat_rspec=$tr_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $tr_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$tr_utt2uttid \
        --repl_repr_utt --repl_utt_set_name=uttid \
        --repl_utt_repr_spec=$tr_repr_wspec --repl_utt_list=$tr_repl_utt_label \
        --repl_utt_id_map=$tr_utt2uttid --repl_utt_wspec=$tr_repl_utt_wspec \
        --repl_utt_img_dir=$tr_repl_utt_img_dir || exit 1;
fi

if [ $stage -le 4 ]; then
    echo "$0: stage 4, traversing latent space ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest --trav --trav_img_dir=$trav_img_dir || exit 1;
fi
