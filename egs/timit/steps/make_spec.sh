# !/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

fs=16000
fn=400
sn=160
fft=400

. parse_options.sh || exit 1;

echo "$0 $@"
if [ $# -ne 3 ]; then
    echo "Usage: $0: [options] <kaldi_src_data> <kaldi_tar_data> <egs_tar_data>"
    echo "options:"
    echo "  --fs <fs>                   # sampling rate"
    echo "  --fn <fn>                   # frame size in sample"
    echo "  --sn <sn>                   # shift size in sample"
    echo "  --fft <fft>                 # fft size"
    exit 1;
fi

kaldi_src_data=$1       # abs path to kaldi_egs_dir
egs_wav_data=$2         # abs path
egs_spec_scp_data=$3    # abs path

if [ ! -f $kaldi_src_data/wav.scp ]; then
    echo "wav.scp not found in ${kaldi_src_data}. run kaldi scripts first"
    exit 1;
fi 

if [ ! -d $egs_spec_scp_data ]; then
    mkdir -p $egs_wav_data $egs_spec_scp_data
    cp $kaldi_src_data/* $egs_spec_scp_data || true
    rm -f $egs_spec_scp_data/{cmvn,feats}.scp
    mv $egs_spec_scp_data/wav.scp $egs_spec_scp_data/wav_old.scp

    python src/tools/kaldi/sph_scp_to_wav.py \
        $egs_spec_scp_data/wav_old.scp \
        $egs_wav_data $egs_spec_scp_data/wav.scp || exit 1;
    rm $egs_spec_scp_data/wav_old.scp

    feat_opts="{\"decom\":\"mp\", \"frame_size_n\":$fn,"
    feat_opts="$feat_opts \"shift_size_n\":$sn, \"fft_size\":$fft,"
    feat_opts="$feat_opts \"awin\":None, \"log_floor\":-20}"

    python_cmd="from tools.kaldi.prep_kaldi_feat import dump_kaldi_spec;"
    python_cmd="$python_cmd dump_kaldi_spec(\"$egs_spec_scp_data/wav.scp\"," 
    python_cmd="$python_cmd \"$egs_spec_scp_data/feats\", $feat_opts)"
    python -c "$python_cmd" || exit 1;
fi
