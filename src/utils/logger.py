from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

_DEBUG_FLAG = False
_LOGGER = None

def set_debug_flag(val):
    assert(isinstance(val, bool))
    global _DEBUG_FLAG
    _DEBUG_FLAG = val

def set_logger(logger):
    assert(isinstance(logger, custom_logger))
    global _LOGGER
    _LOGGER = logger

def unset_logger():
    global _LOGGER
    if _LOGGER is not None:
        _LOGGER.close()
        _LOGGER = None

class custom_logger(object):
    def __init__(self, log_path='./log', formatter_str=None, debug=None):
        print("setting logger and file handler (%s)" % log_path)
        self.log_path = log_path

        if not os.path.isdir(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))

        if formatter_str is None:
            self.formatter_str = "%(process)d %(message)s"
        else:
            self.formatter_str = formatter_str

        self._debug = _DEBUG_FLAG if debug is None else debug
    
        self._logger = logging.getLogger()
        self._handler = logging.FileHandler(log_path)
        self._formatter = logging.Formatter(self.formatter_str)
        
        self._handler.setFormatter(self._formatter)
        self._logger.addHandler(self._handler)

        if self._debug:
            self._logger.setLevel(logging.DEBUG)
            self._handler.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)
            self._handler.setLevel(logging.INFO)

    def __enter__(self):
        return self

    def info(self, *msg):
        self._logger.info(*msg)

    def debug(self, *msg):
        self._logger.debug(*msg)

    def __exit__(self, exc_type, exc_value, traceback):
        print("removing handler (%s) now..." % self.log_path)
        self._logger.removeHandler(self._handler)

    def close(self):
        print("removing handler (%s) now..." % self.log_path)
        self._logger.removeHandler(self._handler)

def info(*msg):
    if _LOGGER is not None:
        _LOGGER.info(*msg)
    print(*msg)

def debug(*msg):
    if _LOGGER is not None:
        _LOGGER.debug(*msg)
    if _DEBUG_FLAG:
        print(*msg)
