#!/bin/bash
source_path="/home/xiangyu/Common/EUVS_data/Level_3_MARS/loc_06_case_1"

# source_path="/media/xiangyu/2d84fb22-ea3e-4235-9526-070f01ac833a/EUVS_data/Level_3_MARS/loc_06_case_1/models/3DGS"
model_path="$source_path/models/3DGS"

python metrics_with_dyn_masks.py -s "$source_path" -m "$model_path" -e "all"