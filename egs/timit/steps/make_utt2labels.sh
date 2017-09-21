# !/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

isutt=true

. parse_options.sh || exit 1;

echo "$0 $@"
if [ $# -ne 2 ] && [ $# -ne 3 ] ; then
    echo "Usage: $0: [options] <utt2lab-or-uttlist> <utt2labid> <lab2labid>"
    echo "options:"
    echo "  --isutt <true|false>        # if label is utterance itself"
    exit 1;
fi

utt2lab=$1
utt2labid=$2
lab2labid=$3

if $isutt; then
    if [ ! -f $utt2labid ]; then
        tmp_ids=$(mktemp -t tmp_ids.XXXX)
        tmp_utts=$(mktemp -t tmp_utts.XXXX)
        seq 1 $(wc -l $utt2lab | awk '{ print $1 }') > $tmp_ids
        cat $utt2lab | awk '{ print $1 }' > $tmp_utts
        paste -d" " $tmp_utts $tmp_ids > $utt2labid
        rm $tmp_ids $tmp_utts
    fi
else
    if [ ! -f $utt2labid ] || [ ! -f $lab2labid ]; then
        python src/tools/kaldi/prep_utt2label.py \
            $utt2labid $lab2labid $utt2lab || exit 1;
    fi
fi
