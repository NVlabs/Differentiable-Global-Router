# run different parameters for result analysis
data="ispd18_test5_metal5"
python3 main.py --data_path /home/scratch.rliang_hardware/wli1/cu-gr/run/$data.pt --pattern_level 1 --use_gumble True --iter 500 --select_threshold 0.9
