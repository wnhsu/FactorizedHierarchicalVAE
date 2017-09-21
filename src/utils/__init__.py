import os
import shutil
from logger import *

def check_and_makedirs(dir_path):
    if bool(dir_path) and not os.path.exists(dir_path):
        info("creating directory %s" % dir_path)
        os.makedirs(dir_path)

def maybe_copy(src, dest):
    if os.path.exists(dest):
        info("%s exists, using that" % dest)
    else:
        shutil.copyfile(src, dest)
