#!/bin/bash
source_path="path/to/your/data"
model_path="$source_path/models/3DGS"

python metrics_with_dyn_masks.py -s "$source_path" -m "$model_path" -e "all"