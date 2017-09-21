"""Base Dataset Class"""
import os
import cPickle

from datasets import *

class BaseDataset(object):
    """
    Abstract class for dataset
    """
    def __init__(self, **kwargs):
        self._set_conf(**kwargs)
        
        self._load_data()
        self._init_data_plan()
        self._init_normalizer()

    def _set_conf(self, **kwargs):
        """
        set Dataset configurations
        """
        raise NotImplementedError

    def _load_data(self):
        """
        load data into memory
        """
        raise NotImplementedError

    def _init_data_plan(self):
        """
        initialize the data plan
        """
        raise NotImplementedError

    def _make_data_plan(self, if_rand):
        """
        make a data plan for iterator/next_batch
        """
        raise NotImplementedError
    
    def _init_normalizer(self):
        """
        set normalization params
        """
        if self._mvn_path:
            if os.path.exists(self._mvn_path):
                with open(self._mvn_path) as f:
                    info("loading mvn params from %s" % self._mvn_path)
                    self._mvn_params = cPickle.load(f)
            else:
                self._compute_mvn_and_save(self._mvn_path)
        else:
            self._mvn_params = None

    def _compute_mvn_and_save(self, path):
        """
        compute mean/variance and save in pkl format to path
        """
        raise NotImplementedError

    def iterator(self, bs=256):
        raise NotImplementedError

    def next_batch(self, bs=256):
        raise NotImplementedError

    def get_item(self, index):
        raise NotImplementedError

    def apply_mvn(self, batch):
        raise NotImplementedError

    def undo_mvn(self, batch):
        raise NotImplementedError

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def feats(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def feat_shape(self):
        raise NotImplementedError

    @property
    def feat_dim(self):
        raise NotImplementedError
