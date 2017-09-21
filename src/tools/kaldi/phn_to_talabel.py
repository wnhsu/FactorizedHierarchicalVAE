from __future__ import print_function

def convert_phn_to_talabels(phn_scp, map_file, talabel_path, time_scale=160., num_phones=39):
    """
    INPUTS:
        phn_scp         - path to the file of mapping from utterance to *.phn file, used in TIMIT
        talabel_path    - output path of time-aligned label files used in KaldiRADataset
        time_scale      - ratio of temporal scales between *.phn and target data representation.
                          TIMIT sample rate is 16kHz, while FBank is often sampled at 100Hz; 
                          in this case, time_scale=160 should be used
    
    OUTPUTS:
        None
    """
    phone_to_idx = load_timit_phone_map(map_file, num_phones)
    with open(talabel_path, "w") as f_out, open(phn_scp) as f_in:
        for line in f_in:
            utt_id, utt_phn_path = line.rstrip().split()
            f_out.write("%s\n" % utt_id)
            with open(utt_phn_path) as f_phn:
                for talabel_str in f_phn:
                    toks = talabel_str.rstrip().split()
                    if toks[2] != "q":
                        start = int(float(toks[0]) / time_scale)
                        end = int(float(toks[1]) / time_scale)
                        label_idx = phone_to_idx[toks[2]]
                        f_out.write("%s %s %s\n" % (start, end, label_idx))

def load_timit_phone_map(map_file, num_phones=39):
    """
    INPUTS:
        map_file        - the phones.60-48-39.map file in TIMIT
        num_phones      - [60|48|39], size of phone inventory used. (glottal stop 'q' deleted)

    OUTPUS:
        label2id        - a mapping from 60-phone label (str) to index (int)
    """
    col_idx = _num_phone_to_col_idx(num_phones)
    label2id = dict()
    with open(map_file) as f:
        phone_sets = zip(*[line.rstrip().split() for line in f if line != "q\n"])
    target_phones = phone_sets[col_idx]
    phone_to_target_phone = dict(zip(phone_sets[0], target_phones))

    uniq_target_phones = []
    for p in target_phones:
        if not p in uniq_target_phones:
            uniq_target_phones.append(p)
    assert(len(uniq_target_phones) == num_phones)

    target_phone_to_idx = dict(zip(uniq_target_phones, range(1, 1 + num_phones)))
    phone_to_idx = dict()
    for p in phone_to_target_phone:
        phone_to_idx[p] = target_phone_to_idx[phone_to_target_phone[p]]
    return phone_to_idx

def dump_timit_phone2phoneid(map_file, label2id_path, num_phones=39):
    col_idx = _num_phone_to_col_idx(num_phones)
    with open(map_file) as f:
        phone_sets = zip(*[line.rstrip().split() for line in f if line != "q\n"])
    target_phones = phone_sets[col_idx]

    uniq_target_phones = []
    for p in target_phones:
        if not p in uniq_target_phones:
            uniq_target_phones.append(p)
    assert(len(uniq_target_phones) == num_phones)

    with open(label2id_path, "w") as f:
        for i, p in enumerate(uniq_target_phones):
            f.write("%s %s\n" % (p, i + 1))

def _num_phone_to_col_idx(num_phones):
    if num_phones == 60:
        col_idx = 0
    elif num_phones == 48:
        col_idx = 1
    elif num_phones == 39:
        col_idx = 2
    else:
        raise ValueError("invalid number of phones %s" % num_phones)
    return col_idx

if __name__ == "__main__":
    import argparse
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("phn_scp", help="")
    parser.add_argument("map_file", help="")
    parser.add_argument("talabel_path", help="")
    parser.add_argument("label2id_path", help="")
    parser.add_argument("--time_scale", type=float, default=160., help="")
    parser.add_argument("--num_phones", type=int, default=39, help="")
    args = parser.parse_args()
    
    pp = pprint.PrettyPrinter(indent=4)
    print("phn_to_talabel.py arguments:")
    pp.pprint(vars(args))
    dump_timit_phone2phoneid(args.map_file, args.label2id_path, args.num_phones)
    convert_phn_to_talabels(args.phn_scp, args.map_file, args.talabel_path, args.time_scale, args.num_phones)
