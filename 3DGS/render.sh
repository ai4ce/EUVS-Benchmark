#!/bin/bash
source_path="/home/xiangyu/Common/EUVS_data/Level_1_nuplan/v_loc1_level1_mulitraveral"
model_path="$source_path/models/3DGS"

python render.py -m "$model_path"