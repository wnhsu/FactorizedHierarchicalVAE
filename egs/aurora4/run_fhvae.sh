#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

AURORA4_DIR=../../kaldi/egs/aurora4/s5
feat_type=spec

# input sections
exp_dir=exp/fhvae_lstm1L256_lat32_32_ad0_${feat_type}_20_20_uttid
train_conf=conf/train/fhvae/e500_p50_lr1e-3_bs256_nbs2000_ad0.cfg
model_conf=conf/model/fhvae/lstm_1L_256_lat_32_32.cfg
dataset_conf=data/${feat_type}_scp/dev_0330_tr90/dataset.cfg
set_name=uttid

feat_dir=data/${feat_type}_scp
tr=train_si84_clean
dt=dev_0330
tt=test_eval92

fac_label=qual_conf/vis_uttid.txt
trav_label=qual_conf/trav_uttid.txt
dt_denoise_label=qual_conf/repl_denoise.txt
dt_convspk_label=qual_conf/repl_spk.txt

stage=0

. ./path.sh 
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

set -eu

if [ $stage -le -1 ]; then
    ./local/${feat_type}_data_prep.sh --AURORA4_KALDI_EGS $AURORA4_DIR || exit 1;
fi

tr_feat_rspec=scp:$feat_dir/$tr/feats.scp
tr_utt2uttid=$feat_dir/$tr/utt2uttid

dt_feat_rspec=scp:$feat_dir/$dt/feats.scp
dt_utt2uttid=$feat_dir/$dt/utt2uttid

tt_feat_rspec=scp:$feat_dir/$tt/feats.scp
tt_utt2uttid=$feat_dir/$tt/utt2uttid

# output sections
fac_wspec=ark:$exp_dir/eval/fac/test.ark
fac_img_dir=$exp_dir/eval/fac/img

tr_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/train.ark
dt_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/dev.ark
tt_z2_meanvar_wspec=ark:$exp_dir/eval/z2_meanvar/test.ark
tr_repr_wspec=ark:$exp_dir/eval/repr/${tr}.ark
dt_repr_wspec=ark:$exp_dir/eval/repr/${dt}.ark
tt_repr_wspec=ark:$exp_dir/eval/repr/${tt}.ark

dt_denoise_wspec=ark:$exp_dir/eval/repl_utt/denoise/${dt}.ark
dt_denoise_img_dir=$exp_dir/eval/repl_utt/denoise/img

dt_convspk_wspec=ark:$exp_dir/eval/repl_utt/convspk/${dt}.ark
dt_convspk_img_dir=$exp_dir/eval/repl_utt/convspk/img

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
        --feat_rspec=$tr_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $tr_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$tr_utt2uttid \
        --dump_lat --use_mean --use_logvar --dump_z2 \
        --test_lat_wspec=$tr_z2_meanvar_wspec --test_utt_id_map=$tr_utt2uttid \
        --dump_repr --repr_set_name=uttid \
        --test_repr_wspec=$tr_repr_wspec --test_repr_id_map=$tr_utt2uttid || exit 1;
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --feat_rspec=$dt_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $dt_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$dt_utt2uttid \
        --dump_lat --use_mean --use_logvar --dump_z2 \
        --test_lat_wspec=$dt_z2_meanvar_wspec --test_utt_id_map=$dt_utt2uttid \
        --dump_repr --repr_set_name=uttid \
        --test_repr_wspec=$dt_repr_wspec --test_repr_id_map=$dt_utt2uttid || exit 1;
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --feat_rspec=$tt_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $tt_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$tt_utt2uttid \
        --dump_lat --use_mean --use_logvar --dump_z2 \
        --test_lat_wspec=$tt_z2_meanvar_wspec --test_utt_id_map=$tt_utt2uttid \
        --dump_repr --repr_set_name=uttid \
        --test_repr_wspec=$tt_repr_wspec --test_repr_id_map=$tt_utt2uttid || exit 1;
fi

if [ $stage -le 3 ]; then
    echo "$0: stage 3, replacing utterance s-vector ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --feat_rspec=$dt_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $dt_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$dt_utt2uttid \
        --repl_repr_utt --repl_utt_set_name=uttid \
        --repl_utt_repr_spec=$dt_repr_wspec --repl_utt_list=$dt_denoise_label \
        --repl_utt_id_map=$dt_utt2uttid --repl_utt_wspec=$dt_denoise_wspec \
        --repl_utt_img_dir=$dt_denoise_img_dir || exit 1;
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest \
        --feat_rspec=$dt_feat_rspec --feat_set_name=uttid \
        --feat_label_N=$(($(wc -l $dt_utt2uttid | awk '{print $1}') + 1)) \
        --feat_utt2label_path=$dt_utt2uttid \
        --repl_repr_utt --repl_utt_set_name=uttid \
        --repl_utt_repr_spec=$dt_repr_wspec --repl_utt_list=$dt_convspk_label \
        --repl_utt_id_map=$dt_utt2uttid --repl_utt_wspec=$dt_convspk_wspec \
        --repl_utt_img_dir=$dt_convspk_img_dir || exit 1;
fi

if [ $stage -le 4 ]; then
    echo "$0: stage 4, traversing latent space ($(hostname); $(date))"
    python src/scripts/run_nips17_fhvae_exp.py \
        --exp_dir=$exp_dir --notest --trav \
        --trav_img_dir=$trav_img_dir --trav_spec=$trav_label || exit 1;
fi
