import numpy as np

class Segment(object):
    def __init__(self, utt_id, start, end, label=None):
        self._utt_id = utt_id
        self._start = start
        self._end = end
        self._label = label

    def __str__(self):
        return "(%s, %s, %s, %s)" % (self.utt_id, self.start, self.end, self.label)

    def __repr__(self):
        return self.__str__()

    @property
    def utt_id(self):
        return self._utt_id

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def label(self):
        return self._label

def make_seg_list(
        utt_index_list, utt_list, utt_len_list, 
        seg_len, seg_shift, if_seg_rand, utt2label=None):
    """
    INPUTS:
        if_seg_rand     - randomize segmentation within an utterance
    
    OUTPUTS:
        seg_list        - list of Segment objects, start_f is starting frame; 
                          end_f is ending frame; label is None utt2label is None
    """
    seg_list = []
    for utt_index in utt_index_list:
        utt_id = utt_list[utt_index]
        utt_len = utt_len_list[utt_index]
        label = utt2label[utt_id] if utt2label else None
        n_segs = (utt_len - seg_len) // seg_shift + 1
        if if_seg_rand:
            start_f_list = np.random.choice(xrange(utt_len - seg_len + 1), n_segs)
        else:
            start_f_list = np.arange(n_segs) * seg_shift
        for f in start_f_list:
            seg_list.append(Segment(utt_id, f, f + seg_len, label))
    return seg_list

def make_talabel_seg_list(
        utt_index_list, utt_list, utt_len_list, seg_len, utt2talabels):
    """
    INPUTS:
    
    OUTPUTS:
        seg_list        - list of Segment objects centered at time-aligned labels.
                          start_f is starting frame; end_f is ending frame;
    """
    seg_list = []
    for utt_index in utt_index_list:
        utt_id = utt_list[utt_index]
        utt_len = utt_len_list[utt_index]
        utt_segs = [talabel.get_centered_seg(seg_len, max_t=utt_len) \
                for talabel in utt2talabels[utt_id]]
        seg_list += [Segment(utt_id, seg[0], seg[1], seg[2]) \
                for seg in utt_segs if seg is not None]
    return seg_list
