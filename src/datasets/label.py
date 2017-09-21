class TimeAlignedLabel(object):
    """class for time-aligned label"""
    def __init__(self, label, start_t, end_t):
        self._label = label
        self._start_t = start_t
        self._end_t = end_t
    
    def __str__(self):
        return "(%s, %s, %s)" % (self.label, self.start_t, self.end_t)

    def __repr__(self):
        return self.__str__()

    @property
    def label(self):
        return self._label

    @property
    def start_t(self):
        return self._start_t

    @property
    def end_t(self):
        return self._end_t
    
    @property
    def center_t(self):
        return int(round((self.start_t + self.end_t) / 2.))

    @property
    def duration(self):
        return self.end_t - self.start_t + 1

    def get_centered_seg(self, seg_len, min_t=0, max_t=None):
        """
        return segment start/end index if not exceeding valid range; 
        max_t is usually set as len(seq).
        """
        seg_start_t = self.center_t - int(round(seg_len / 2.))
        seg_end_t = seg_start_t + seg_len
        if min_t is not None and seg_start_t < min_t:
            return None
        elif max_t is not None and seg_end_t > max_t:
            return None
        else:
            return (seg_start_t, seg_end_t, self.label)

def load_time_aligned_labels(path):
    """
    INPUTS:
        path            - time-aligned label file. formatted as uttid line 
                          followed by "start_f end_f label" lines.
                          start_f, end_f, and label are int
    OUTPUTS:
        utt_to_talabels  - a mapping from uttid to a list of TimeAlignedLabel
    """
    utt_to_talabels = dict()
    with open(path) as f:
        uttid = ""
        ta_label_list = []
        for line in f:
            toks = line.rstrip().split()
            if len(toks) == 1:
                if uttid:
                    utt_to_talabels[uttid] = ta_label_list
                uttid = toks[0]
                ta_label_list = []
            else:
                assert(len(toks) == 3)
                ta_label_list.append(
                        TimeAlignedLabel(int(toks[2]), int(toks[0]), int(toks[1])))

        if uttid:
            utt_to_talabels[uttid] = ta_label_list

    return utt_to_talabels


def load_label(path):
    """
    INPUTS:
        path            - label file. formatted as "uttid label" per line. label is int
    
    OUTPUTS:
        utt_to_label    - a mapping from uttid to a int
    """
    with open(path) as f:
        toks = [line.rstrip().split() for line in f]
    utt_to_label = dict([(tok[0], int(tok[1])) for tok in toks])
    return utt_to_label
