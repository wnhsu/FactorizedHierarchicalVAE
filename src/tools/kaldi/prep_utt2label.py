"""
generate utt2labelid and label_id_map from utt2label of the following format:
    utt1 label1
    utt2 label2
"""
from __future__ import print_function
import numpy as np
from collections import OrderedDict

def prep_utt2label(utt2labelid_path, label_id_map_path, utt2label_paths):
    print("generating utt2labelid(%s) and label_id_map(%s)" % (
            utt2labelid_path, label_id_map_path) + \
            " from utt2labels(%s)" % (utt2label_paths))
    utt_list = []
    label_list = []
    for utt2label_path in utt2label_paths:
        with open(utt2label_path) as f:
            _utt_list, _label_list = zip(*[line.rstrip().split() for line in f])
            utt_list += _utt_list
            label_list += _label_list
    if len(utt_list) != len(np.unique(utt_list)):
        raise ValueError("duplicated utt detected! check %s first" % (
                utt2label_paths,))
    utt2label = OrderedDict(zip(utt_list, label_list))

    _, idx = np.unique(utt2label.values(), return_index=True)
    unique_labels = np.array(utt2label.values())[np.sort(idx)]

    label_id_map = dict(zip(unique_labels, np.arange(len(unique_labels)) + 1))
    with open(label_id_map_path, "w") as f:
        for k in unique_labels:
            f.write("%s %s\n" % (k, label_id_map[k]))
    with open(utt2labelid_path, "w") as f:
        for utt, label in utt2label.iteritems():
            f.write("%s %s\n" % (utt, label_id_map[label]))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("utt2labelid_path")
    parser.add_argument("label_id_map_path")
    parser.add_argument("utt2label_paths", nargs="+")
    args = parser.parse_args()
    prep_utt2label(args.utt2labelid_path, args.label_id_map_path, args.utt2label_paths)
