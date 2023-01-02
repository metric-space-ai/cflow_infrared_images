#!/bin/bash
export PATH="/home/nvidia/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate test
cd /home/nvidia/Documents/precon
python server.py

