import cv2 as cv
import os
import numpy as np
import torch
import glob
import scipy.io as sio

"""
Normal coordinates for DiLiGenT dataset: (same for light direction)
    y    
    |   
    |  
    | 
     --------->   x
   /
  /
 z 
x direction is looking right
y direction is looking up
z direction is looking outside the image

we convert it to :
Normal coordinates for DiLiGenT dataset: (same for light direction)
     --------->   x
    |   
    |  
    | 
    y    
x direction is looking right
y direction is looking down
z direction is looking into the image

"""


def parse_txt(filename):
    out_list = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        for x in lines:
            lxyz = np.array([float(v) for v in x.strip().split()], dtype=np.float32)
            out_list.append(lxyz)
    out_arr = np.stack(out_list, axis=0).astype(np.float32)
    return out_arr


def read_mat_file(filename):
    """
    :return: Normal_ground truth in shape: (height, width, 3)
    """
    mat = sio.loadmat(filename)
    gt_n = mat['Normal_gt']
    return gt_n.astype(np.float32)


def load_diligent(path, cfg=None):
    images = []
    for img_file in sorted(glob.glob(os.path.join(path,"[0-9]*.png"))):
        img = cv.imread(img_file)[:,:,::-1].astype(np.float32) / 255.
        # img = (img - 0.5) * 2
        images.append(img)
    images = np.stack(images, axis=0)

    mask_files = os.path.join(path, "mask.png")
    mask = cv.imread(mask_files, 0).astype(np.float32) / 255.

    light_dir_files = os.path.join(path, "light_directions.txt")
    light_dir = parse_txt(light_dir_files)
    light_dir[..., 1:] = -light_dir[..., 1:]  # convert y-> -y   z->-z

    light_intensity_files = os.path.join(path, "light_intensities.txt")
    light_intensity = parse_txt(light_intensity_files)

    gt_normal_files = os.path.join(path, "Normal_gt.mat")
    gt_normal = read_mat_file(gt_normal_files)
    gt_normal[..., 1:] = -gt_normal[..., 1:]  # convert y-> -y   z->-z

    if os.path.basename(path) == 'bearPNG':
        images = images[20:]
        light_dir = light_dir[20:]
        light_intensity = light_intensity[20:]

    if hasattr(cfg.dataset, 'sparse_input_random_seed') and hasattr(cfg.dataset, 'sparse_input'):
        if cfg.dataset.sparse_input_random_seed is not None and cfg.dataset.sparse_input is not None:
            np.random.seed(cfg.dataset.sparse_input_random_seed)
            select_idx = np.random.permutation(len(images))[:cfg.dataset.sparse_input]
            print('Random seed: %d .   Selected random index: ' % cfg.dataset.sparse_input_random_seed, select_idx)
            images = images[select_idx]
            light_dir = light_dir[select_idx]
            light_intensity = light_intensity[select_idx]

    out_dict = {'images': images, 'mask': mask, 'light_direction': light_dir, 'light_intensity': light_intensity, 'gt_normal': gt_normal}
    return out_dict


def load_unitsphere():
    mask_files = os.path.join("./data/DiLiGenT/pmsData/ballPNG", "mask.png")
    mask = cv.imread(mask_files, 0).astype(np.float32) / 255.
    gt_normal_files = os.path.join("./data/DiLiGenT/pmsData/ballPNG", "Normal_gt.mat")
    gt_normal = read_mat_file(gt_normal_files)
    gt_normal[..., 1:] = -gt_normal[..., 1:]  # convert y-> -y   z->-z

    return {'mask': mask, 'gt_normal': gt_normal}
