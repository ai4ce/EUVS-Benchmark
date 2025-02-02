#!/bin/bash
source_path="/home/xiangyu/Common/EUVS_data/Level_1_nuplan/v_loc1_level1_mulitraveral"
model_path="$source_path/models/3DGS"

python train.py -s "$source_path" -m "$model_path" --method "masked_3dgs"


##### Depth map regularization #####
# source_path=/home/xiangyu/Common/EUVS_data/Level3_new/loc06
# model_path="$source_path/models/3DGS"
# depth_map_path="$source_path/depth"

# python train.py -s "$source_path" -m "$model_path" --method "masked_3dgs" -d "$depth_map_path"