# !/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

cv_percent=10

. parse_options.sh || exit 1;

echo "$0 $@"
if [ $# -ne 2 ]; then
    echo "Usage: $0: [options] <kaldi_egs_dir> <data_dir>"
    echo "options:"
    echo "  --cv_percent <percent>      # percentage of the validation set"
    exit 1;
fi

kaldi_egs_dir=$1    # abs path
data_dir=$2         # abs path

cd $kaldi_egs_dir
. ./cmd.sh
. ./path.sh

data_dir=$(echo $data_dir | sed 's#/*$##')  # remove trailing '/'
tr_percent=$((100 - cv_percent))

tmp_utt_list=$(mktemp)
tmp_utt_list2=$(mktemp)
tmp_cv_utt_list=$(mktemp)
tmp_tr_utt_list=$(mktemp)

awk '{print $1}' $data_dir/feats.scp > $tmp_utt_list
n_utt=$(wc -l $tmp_utt_list | awk '{print $1}')
n_cv_utt=$(( n_utt * cv_percent / 100 ))
n_tr_utt=$(( n_utt - n_cv_utt ))

shuf $tmp_utt_list > $tmp_utt_list2
head -n $n_cv_utt $tmp_utt_list2 > $tmp_cv_utt_list
tail -n $n_tr_utt $tmp_utt_list2 > $tmp_tr_utt_list

utils/subset_data_dir.sh --utt-list $tmp_cv_utt_list \
    $data_dir ${data_dir}_cv${cv_percent} || exit 1;
utils/subset_data_dir.sh --utt-list $tmp_tr_utt_list \
    $data_dir ${data_dir}_tr${tr_percent} || exit 1;

rm -f $tmp_utt_list $tmp_utt_list2 $tmp_cv_utt_list $tmp_tr_utt_list || true;
cd -
