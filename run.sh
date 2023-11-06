#!/bin/bash
#conda init bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate animatediff
touch out.log
ps aux | grep animate_diff_app.py | grep -v grep | awk '{print $2}' | xargs kill -9
export CUDA_VISIBLE_DEVICES=4 && python3 animatediff_app.py
export CUDA_VISIBLE_DEVICES=4 && python3 animatediff_app_v1.py
