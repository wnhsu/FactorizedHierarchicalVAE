import numpy as np
from collections import namedtuple

def make_one_hot(labels, n_class):
    assert(np.asarray(labels).ndim == 1)
    one_hot = np.zeros((len(labels), n_class), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.
    return one_hot

class IndexedDict(object):
    def __init__(self, keys, vals):
        _IndexedDict = namedtuple("_IndexedDict", keys)
        self._label_sets = _IndexedDict(*vals)
        self._keys = keys
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._label_sets[key]
        elif isinstance(key, str):
            return getattr(self._label_sets, key)
        else:
            raise ValueError("invalid key %s" % key)

    def keys(self):
        return self._keys

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys
