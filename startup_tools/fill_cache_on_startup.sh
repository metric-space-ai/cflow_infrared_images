eval "$(conda shell.bash hook)"
conda activate test
cd /home/nvidia/Documents/precon
python client.py -i /home/nvidia/Documents/precon/cache_input.tiff
