import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="car. tire.")
parser.add_argument("--input-dir", default="input_images")
parser.add_argument("--output-dir", default="output_masks")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
INPUT_DIR = args.input_dir
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)
DUMP_JSON_RESULTS = not args.no_dump_json

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment settings
# Use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

# Get list of images in input directory
image_files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

# Loop over each image
for image_file in image_files:
    img_path = os.path.join(INPUT_DIR, image_file)
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    sam2_predictor.set_image(image_np)

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()
    class_names = results[0]["labels"]

    if len(input_boxes) == 0:
        # No objects detected, output an all-zero mask
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        output_mask_path = OUTPUT_DIR / f"{Path(image_file).stem}.png"
        cv2.imwrite(str(output_mask_path), mask)
        print(f"No objects detected in {image_file}. Saved all-zero mask.")
        continue

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Combine all masks into one
    combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for mask in masks:
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.width, image.height))
        combined_mask = np.logical_or(combined_mask, mask_resized).astype(np.uint8)

    # Set mask pixels to 1
    combined_mask[combined_mask > 0] = 1
    final_mask = 1 - combined_mask
    # Save the mask as PNG
    output_mask_path = OUTPUT_DIR / f"{Path(image_file).stem}.png"
    cv2.imwrite(str(output_mask_path), final_mask * 255)

    print(f"Processed {image_file}, mask saved to {output_mask_path}")
