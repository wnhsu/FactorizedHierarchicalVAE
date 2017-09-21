# !/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

hasspk=true
hasphone=false
hasstft=false

egs=timit
seg_len=20
seg_shift=20
seg_rand=True
feat_type=fbank_raw

n_chan=1
use_chan=0
remove_0th=False
max_to_load=-1
if_rand=True

decom=
fs=
fn=
sn=
fft=

. parse_options.sh || exit 1;

echo "$0 $@"
if [ $# -ne 4 ]; then
    echo "Usage: $0: [options] <tr_dir> <dt_dir> <tt_dir> <dataset_cfg>"
    echo "options:"
    echo "  --hasspk <true|false>       # if utt2spkid/spk2spkid is prepared"
    echo "  --hasphone <true|false>     # if utt2phoneid/phone2phoneid is prepared"
    echo "  --hasstft <true|false>      # if [stft] section exists"
    echo "  --seg_len <seg_len>         # segment length"
    echo "  --seg_shift <seg_shift>     # segment shift"
    echo "  --seg_rand <True|False>     # if randomly generate segments"
    echo "  --feat_type <feat_type>     # feature type"
    echo "  --n_chan <n_chan>           # number of channels"
    echo "  --use_chan <use_chan>       # channel indexes to use, separated by ,"
    echo "  --remove_0th <remove_0th>   # if remove the 0-th dimension (DC)"
    echo "  --max_to_load <max_to_load> # maximum utterance to load into memory"
    echo "  --if_rand <if_rand>         # if randomize utterance order"
    echo "  --decom <mp|ri>             # decomposition for spectrogram features"
    echo "  --fs <fs>                   # sampling rate"
    echo "  --fn <fn>                   # frame size in sample"
    echo "  --sn <sn>                   # shift size in sample"
    echo "  --fft <fft>                 # fft size"
    exit 1;
fi

tr_dir=$1
dt_dir=$2
tt_dir=$3
dataset_cfg=$4

for d in $tr_dir $dt_dir $tt_dir; do
    for f in $d/{feats.scp,utt2uttid}; do
        if [ ! -f $f ]; then
            echo "$0: $f not found" && exit 1;
        fi
    done
    if $hasspk; then
        for f in $d/{spk2spkid,utt2spkid}; do
            if [ ! -f $f ]; then
                echo "$0: $f not found" && exit 1;
            fi
        done
    fi
done

n_tr_utt=$(wc -l ${tr_dir}/utt2uttid | awk '{ print $1 }')
n_dt_utt=$(wc -l ${dt_dir}/utt2uttid | awk '{ print $1 }')
n_tt_utt=$(wc -l ${tt_dir}/utt2uttid | awk '{ print $1 }')

tr_utt2label_paths="uttid:$((n_tr_utt + 1)):$tr_dir/utt2uttid"
dt_utt2label_paths="uttid:$((n_dt_utt + 1)):$dt_dir/utt2uttid"
tt_utt2label_paths="uttid:$((n_tt_utt + 1)):$tt_dir/utt2uttid"

if $hasspk; then
    n_tr_spk=$(wc -l ${tr_dir}/spk2spkid | awk '{ print $1 }')
    n_dt_spk=$(wc -l ${dt_dir}/spk2spkid | awk '{ print $1 }')
    n_tt_spk=$(wc -l ${tt_dir}/spk2spkid | awk '{ print $1 }')

    tr_utt2label_paths="$tr_utt2label_paths,spk:$((n_tr_spk + 1)):$tr_dir/utt2spkid"
    dt_utt2label_paths="$dt_utt2label_paths,spk:$((n_dt_spk + 1)):$dt_dir/utt2spkid"
    tt_utt2label_paths="$tt_utt2label_paths,spk:$((n_tt_spk + 1)):$tt_dir/utt2spkid"
fi

if $hasphone; then
    n_tr_phone=$(wc -l ${tr_dir}/phone2phoneid | awk '{ print $1 }')
    n_dt_phone=$(wc -l ${dt_dir}/phone2phoneid | awk '{ print $1 }')
    n_tt_phone=$(wc -l ${tt_dir}/phone2phoneid | awk '{ print $1 }')

    tr_utt2talabels_paths="phone:$((n_tr_phone + 1)):$tr_dir/utt2phoneid.talabel"
    dt_utt2talabels_paths="phone:$((n_dt_phone + 1)):$dt_dir/utt2phoneid.talabel"
    tt_utt2talabels_paths="phone:$((n_tt_phone + 1)):$tt_dir/utt2phoneid.talabel"
fi


        tee $dataset_cfg <<EOF
[data]
fmt                         = kaldi_ra
egs                         = $egs
train_feat_rspec            = scp:${tr_dir}/feats.scp
dev_feat_rspec              = scp:${dt_dir}/feats.scp
test_feat_rspec             = scp:${tt_dir}/feats.scp
train_utt2label_paths       = $tr_utt2label_paths
dev_utt2label_paths         = $dt_utt2label_paths
test_utt2label_paths        = $tt_utt2label_paths
train_utt2talabels_paths    = $tr_utt2talabels_paths
dev_utt2talabels_paths      = $dt_utt2talabels_paths
test_utt2talabels_paths     = $tt_utt2talabels_paths
mvn_path                    = ${tr_dir}/mvn.pkl

seg_len         = $seg_len
seg_shift       = $seg_shift
seg_rand        = $seg_rand

n_chan          = $n_chan
use_chan        = $use_chan
remove_0th      = $remove_0th
max_to_load     = $max_to_load
if_rand         = $if_rand

feat_type       = $feat_type

EOF

if $hasstft; then
    tee -a $dataset_cfg<<EOF
decom           = $decom

[stft]
fs              = $fs
frame_size_n    = $fn
shift_size_n    = $sn
fft_size        = $fft
EOF
fi
