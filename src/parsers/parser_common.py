from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from functools import wraps
from ConfigParser import SafeConfigParser, ConfigParser

class base_parser(object):
    def __init__(self, config_path):
        raise NotImplementedError()
    
    def get_config(self):
        raise NotImplementedError()
    
    @staticmethod
    def write_config(config, f):
        raise NotImplementedError()

def use_default(dtype):
    def _use_default(fn):
        @wraps(fn)
        def _wrapped(self, sec, key, val=None, strict=True):
            assert(not val or isinstance(val, dtype))
            if self.has_option(sec, key):
                return fn(self, sec, key)
            elif val is not None or not strict:
                return val
            else:
                raise ValueError("key not found and val is not provided")
        return _wrapped
    return _use_default

class DefaultConfigParser(SafeConfigParser):
    def __init__(self, **kwargs):
        SafeConfigParser.__init__(self, **kwargs)
    
    @use_default(str)
    def get(self, sec, key):
        return SafeConfigParser.get(self, sec, key)

    @use_default(float)
    def getfloat(self, sec, key):
        return SafeConfigParser.getfloat(self, sec, key)
    
    @use_default(int)
    def getint(self, sec, key, val=None, strict=True):
        return SafeConfigParser.getint(self, sec, key)

    @use_default(bool)
    def getboolean(self, sec, key, val=None, strict=True):
        return SafeConfigParser.getboolean(self, sec, key)
