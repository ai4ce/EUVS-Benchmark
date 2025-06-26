#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr_mask, ssim_mask, lpips_mask
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import UnidentifiedImageError
from torchvision.transforms import ToPILImage

# Load model
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model.eval().to("cuda:0")

# Preprocess images
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),
])

def prepare_dino_mask(mask_pil, device):
    # Preprocess
    mask_resized = tf.resize(mask_pil, [518], interpolation=Image.NEAREST)
    mask_cropped = tf.center_crop(mask_resized, [518, 518])

    # Convert to [518,518]
    mask_tensor = mask_cropped.squeeze(0).squeeze(0) > 0.5  # [518,518]
    mask_tensor = mask_tensor.float().to(device)

    # Visible portion
    patches = mask_tensor.unfold(0, 14, 14).unfold(1, 14, 14)  # [37,37,14,14]
    patch_counts = patches.sum(dim=(-1,-2))                    # [37,37]
    patch_weights = patch_counts / (14*14)                     # [37,37]
    
    return patch_weights.flatten()  # [1369]

def calculate_weighted_cossim(feat_render, feat_gt, patch_weights):
    # Per patch cos sim [1,1369]
    cos_per_patch = torch.nn.CosineSimilarity(dim=2)(feat_render, feat_gt).mean()

    # Normalize
    total_weight = patch_weights.sum()
    if total_weight > 1e-6:
        return (cos_per_patch * patch_weights).sum() / total_weight
    return torch.tensor(0.0)  # all mask case

# sorting function
def extract_number(file_path):
    fname = file_path[0].name  
    numbers = ''.join(filter(str.isdigit, fname))  
    return int(numbers) if numbers else -1  

def open_image_with_fallback(path):
    try:
        return Image.open(path)
    except FileNotFoundError:
        try:
            return Image.open(path.with_suffix('.png'))
        except FileNotFoundError:
            raise FileNotFoundError(f"No image found at {path} or {path.with_suffix('.png')}")
        except UnidentifiedImageError:
            raise UnidentifiedImageError(f"Cannot identify image file {path.with_suffix('.png')}")
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Cannot identify image file {path}")


def readMaskWTxt(dm_path, eval_set_path):
    evalset_files = []

    try:
        with open(eval_set_path, 'r', encoding='utf-8') as file:
            evalset_files = [line.strip().replace('.jpg', '.png') for line in file.readlines()]
    except Exception as e:
        print(f"An error occurred while reading {eval_set_path}: {e}")
        return []
    
    files = [(dm_path / fname) for fname in os.listdir(dm_path)]
    
    # Filter out files of interest
    filtered_files = [file for file in files if file.name in evalset_files]
    filtered_files = sorted(filtered_files)
    return filtered_files

def visualize(render, gt, mask, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rendered Image
    axes[0].imshow(render)
    axes[0].set_title('Rendered Image')
    axes[0].axis('off')
    
    # Ground Truth Image
    axes[1].imshow(gt)
    axes[1].set_title('Ground Truth Image')
    axes[1].axis('off')
    
    # Mask Image
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def readImages(renders_dir, gt_dir):
    image_paths = [(renders_dir / fname, gt_dir / fname) for fname in os.listdir(renders_dir)]
    sorted_image_paths = sorted(image_paths, key=extract_number)
    return sorted_image_paths

def evaluate(model_paths, source_paths, eval_type, VISUALIZE):
    full_dict = {}
    per_view_dict = {}
    device = torch.device("cuda:0")
    lpips_model = lpips.LPIPS(net='alex').to(device)

    for scene_dir, source_path in zip(model_paths, source_paths):
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        if eval_type == "all":
            eval_sets = ["train","test"]
        elif eval_type == "test":
            eval_sets = ["test"]
        elif eval_type == "train":
            eval_sets = ["train"]
        else:
            assert "Unrecgnized evaluate type!"
        for eval_set in eval_sets:
            if eval_set == "test":
                test_dir = Path(scene_dir) / "test"
                dm_path = Path(source_path) / "dynamic_masks"
                eval_set_path = Path(source_path) / "test_set.txt"
                json_path = "/test_set_results_w_mask.json"
            elif eval_set == "train":
                test_dir = Path(scene_dir) / "train"
                dm_path = Path(source_path) / "dynamic_masks"
                eval_set_path = Path(source_path) / "train_set.txt"
                json_path = "/train_set_results_w_mask.json"
            else:
                assert "Unrecgnized evaluate type!"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                dynamic_masks = readMaskWTxt(dm_path,eval_set_path)
                image_paths = readImages(renders_dir, gt_dir)

                ssims, psnrs, lpipss, cos_sims = [], [], [], []

                with torch.no_grad():
                    for i, (render_path, gt_path) in enumerate(tqdm(image_paths, desc="Metric evaluation progress")):
                        # render = Image.open(render_path)
                        # gt = Image.open(gt_path)
                        gt = open_image_with_fallback(gt_path)
                        render = open_image_with_fallback(render_path)
                        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :]
                        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :]
                        dm = np.array(Image.open(dynamic_masks[i])) / 255
                        dm = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions

                        # Match mask size with the image size using interpolation
                        dm = F.interpolate(dm, size=(render_tensor.shape[2], render_tensor.shape[3]), mode='nearest')

                        device = torch.device("cuda:0")
                        dm = dm.to(device)

                        # Move tensors to GPU as needed
                        render_tensor = render_tensor.cuda()
                        gt_tensor = gt_tensor.cuda()

                        ssims.append(ssim_mask(render_tensor[0,:,:,:], gt_tensor[0,:,:,:], dm[0,0,:,:]).item())
                        psnrs.append(psnr_mask(render_tensor[0,:,:,:], gt_tensor[0,:,:,:], dm[0,0,:,:]).item())
                        lpipss.append(lpips_mask(render_tensor[0,:,:,:], gt_tensor[0,:,:,:], dm[0,0,:,:], lpips_model))

                        # Extract feature
                        input_render = transform(render).unsqueeze(0).to(device)
                        input_gt = transform(gt).unsqueeze(0).to(device)
                        feat_render = dino_model.get_intermediate_layers(input_render, n=1)[0][:, 1:, :]
                        feat_gt = dino_model.get_intermediate_layers(input_gt, n=1)[0][:, 1:, :]

                        # Cos sim
                        patch_weights = prepare_dino_mask(dm, device)
                        weighted_cos = calculate_weighted_cossim(feat_render, feat_gt, patch_weights)
                        cos_sims.append(weighted_cos.item())

                        
                        # Visualization
                        if VISUALIZE:
                            save_path = Path(scene_dir) / f"visualization_{i}.png"
                            visualize(render, gt, dm[0, 0, :, :].cpu().numpy(), save_path)

                        # Clear memory
                        del render_tensor, gt_tensor
                        torch.cuda.empty_cache()

                # Update dictionaries
                full_dict[scene_dir][method].update({"SSIM": sum(ssims) / len(ssims),
                                                    "PSNR": sum(psnrs) / len(psnrs),
                                                    "LPIPS": sum(lpipss) / len(lpipss),
                                                    "Cos_Similarity": sum(cos_sims) / len(cos_sims)})
                ssim_val = full_dict[scene_dir][method]["SSIM"]
                psnr_val = full_dict[scene_dir][method]["PSNR"]
                lpips_val = full_dict[scene_dir][method]["LPIPS"]
                cos_val = full_dict[scene_dir][method]["Cos_Similarity"]
                final_score = (
                    0.25 * ssim_val +
                    0.15 * psnr_val +
                    0.35 * (1 - lpips_val) +
                    0.25 * cos_val
                )

                # Final score
                full_dict[scene_dir][method]["Final_Score"] = final_score

            with open(scene_dir + json_path, 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_paths', '-s', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--eval_set', '-e', type=str, default="all")
    parser.add_argument('--visualize', '-v',  action="store_true", help="Enable visualization") 
    args = parser.parse_args()

    evaluate(args.model_paths, args.source_paths, args.eval_set, args.visualize)
