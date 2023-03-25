#!/usr/bin/env bash

cd ~/
rm -f ~/.jupyter/jupyter_hostname ~/.jupyter/jupyter_job.err ~/.jupyter/jupyter_job.out
sbatch -J "Jupyter" -t 5-0 -n 6 --nodes=1 --mem-per-cpu=3G --gpus=1 -o ~/.jupyter/jupyter_job.out -e ~/.jupyter/jupyter_job.err --wrap='hostname > ~/.jupyter/jupyter_hostname; OPENBLAS_NUM_THREADS=1 XDG_RUNTIME_DIR="$TMPDIR" jupyter-notebook --no-browser --NotebookApp.allow_origin="https://colab.research.google.com" --ip=`hostname` --port=8899' 
echo "Waiting for job to start..."
while [ ! -f ~/.jupyter/jupyter_hostname ]; do sleep 1; done
echo "Job started on `cat ~/.jupyter/jupyter_hostname`."
