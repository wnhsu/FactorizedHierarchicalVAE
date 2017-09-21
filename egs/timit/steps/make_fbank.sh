# !/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

nj=40
fbank_conf=

. parse_options.sh || exit 1;

echo "$0 $@"
if [ $# -ne 4 ]; then
    echo "Usage: $0: [options] <kaldi_egs_dir> <kaldi_src_data> <kaldi_tar_data> <egs_tar_data>"
    echo "options:"
    echo "  --nj <nj>                   # number of parallel jobs"
    echo "  --fbank_conf <config_file>  # config passed to compute-fbank-feats (abspath)"
    exit 1;
fi

kaldi_egs_dir=$1        # abs path
kaldi_src_data=$2       # rel path to kaldi_egs_dir
kaldi_tar_data=$3       # rel path to kaldi_egs_dir
egs_tar_data=$4         # abs path

cd $kaldi_egs_dir
. ./cmd.sh
. ./path.sh

if [ ! -f $kaldi_src_data/wav.scp ]; then
    echo "wav.scp not found in ${kaldi_src_data}. run kaldi scripts first"
    exit 1;
fi 

if [ ! -d $kaldi_tar_data ]; then
    mkdir -p $kaldi_tar_data
    cp $kaldi_src_data/* $kaldi_tar_data || true
    rm -f $kaldi_tar_data/{cmvn,feats}.scp

    steps/make_fbank.sh --nj $nj --cmd "$train_cmd" \
        --fbank-config $fbank_conf \
        $kaldi_tar_data $kaldi_tar_data/log fbank || exit 1;
fi

rm -rf $egs_tar_data
mkdir -p $egs_tar_data && cp $kaldi_tar_data/* $egs_tar_data/ || true

cd -
