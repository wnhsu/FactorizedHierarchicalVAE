#!/usr/bin/python 

from __future__ import print_function
import os
import subprocess
from utils import *

def sph_scp_to_wav(sph_scp, wav_dir, wav_scp):
    """
    generate wav files with name %(wav_dir)/%(utt_id).wav
    and wav_scp file of "%(utt_id) %(wav_path)" format
    """
    with open(sph_scp) as f:
        lines = [line.rstrip() for line in f]
        sph_list = [(line.split()[0], line.split()[1:-1]) for line in lines]

    if len(sph_list[0][1]) == 0:
        info("%s already contains a list of wav files." % sph_scp)
        subprocess.check_output(["cp", sph_scp, wav_scp])
        return

    check_and_makedirs(wav_dir)
    if os.path.dirname(wav_scp):
        check_and_makedirs(os.path.dirname(wav_scp))
    with open(wav_scp, "w") as f:
        for utt_id, command in sph_list:
            debug(utt_id, command)
            wav_path = "%s/%s.wav" % (wav_dir, utt_id)
            with open(wav_path, "w") as f_wav:
                subprocess.Popen(command, stdout=f_wav)
            f.write("%s %s\n" % (utt_id, os.path.abspath(wav_path)))

if __name__ == "__main__":
    """
    sys.argv[1]:    sph_scp 
    sys.argv[2]:    wav_dir
    sys.argv[3]:    wav_scp
    """
    import sys
    sph_scp_to_wav(sys.argv[1], sys.argv[2], sys.argv[3])
