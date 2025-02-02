#!/bin/bash

path=/home/xiangyu/Projects/DiffCut/data/test_data
img_path="$path/original_img"
output_path="$path/depth_anything_map"

python /home/xiangyu/Projects/Depth-Anything-V2/run.py \
    --encoder vitl --pred-only --grayscale --img-path "$img_path" --outdir "$output_path"



# python utils/make_depth_scale.py --base_dir "$path" --depths_dir "$output_path"
