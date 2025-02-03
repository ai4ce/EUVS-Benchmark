#!/bin/bash

# Define the path 
paths=(
    "path to the dataset"
)

# Extract masks for the images in the dataset
for path in "${paths[@]}"; do
    images_path="$path/images"
    road_mask_path="$path/dynamic_masks"
    sky_mask_path="$path/sky_masks"
    echo "Processing data from: $path"
    python extract_masks.py --text-prompt "person. rider. car. truck. bus. train. motorcycle. bicycle." --input-dir $images_path --output-dir $road_mask_path
    python extract_masks.py --text-prompt "sky." --input-dir $images_path --output-dir $sky_mask_path

done


 