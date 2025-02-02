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

import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from lpipsPyTorch import lpips


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def psnr_mask(img1, img2, mask):
    mask = mask.to(torch.int64)
    mask = mask.unsqueeze(0).expand_as(img1)

    # Flatten mask
    mask_flat = mask.reshape(-1)
    img1_flat = img1.reshape(-1)
    img2_flat = img2.reshape(-1)

    # Non-zero indices
    nonzero_indices = torch.nonzero(mask_flat).squeeze()

    # Only keep non-zero pixel
    img1_nonzero = torch.index_select(img1_flat, 0, nonzero_indices)
    img2_nonzero = torch.index_select(img2_flat, 0, nonzero_indices)

    # MSE
    mse = ((img1_nonzero - img2_nonzero) ** 2).mean()

    # PSNR
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value

def gaussian_window(size, sigma):
    gauss = torch.Tensor([exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def masked_ssim(img1, img2, mask, window, window_size, channel, size_average=True):
    # Apply mask
    mask = mask.float().unsqueeze(0)
    img1 = img1 * mask
    img2 = img2 * mask
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_mask(img1, img2, mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    if img1.is_cuda:
        mask = mask.cuda(img1.get_device())
    
    return masked_ssim(img1, img2, mask, window, window_size, channel, size_average)

def lpips_mask(image1_tensor, image2_tensor, mask, lpips_model):
    mask = (mask == 1).float()

    mask = mask.to(image1_tensor.device)

    mask_tensor = mask.unsqueeze(0).expand_as(image1_tensor)

    # Apply mask to images
    image1_masked = image1_tensor * mask_tensor
    image2_masked = image2_tensor * mask_tensor

    # Calculate LPIPS similarity for unmasked pixels
    with torch.no_grad():
        similarity_score = lpips_model(image1_masked, image2_masked)
        # similarity_score = lpips(image1_masked, image2_masked, net_type='vgg')

    return similarity_score.item()