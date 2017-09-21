. /data/sls/u/wnhsu/code/release/fhvae_nips17_github/nips17_env/bin/activate 
# cuda-8.0
export LIBRARY_PATH=/data/sls/scratch/yzhang87/cuda-8.0/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/data/sls/scratch/yzhang87/cuda-8.0/lib64:/usr/users/yzhang87/cudnn-5.1/lib64:$LD_LIBRARY_PATH
export KALDI_DIR=/data/sls/u/wnhsu/code/release/fhvae_nips17_github/kaldi
export KALDI_PYTHON_DIR=/data/sls/u/wnhsu/code/release/fhvae_nips17_github/kaldi_python/kaldi-python

export PYTHONPATH=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH
export PATH=$KALDI_DIR/egs/wsj/s5/utils:$PATH

/data/sls/scratch/wnhsu/code/test_slurm.sh || { echo "slurm test failed" && exit 1; }
