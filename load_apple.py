import cv2 as cv
import os
import numpy as np
import glob
import OpenEXR, Imath

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


def get_channel_from_exr(exr, channel_name):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    arr_str = exr.channel(channel_name, pt)
    arr = np.frombuffer(arr_str, dtype=np.float32)
    arr.shape = (size[1], size[0])

    return arr


def data_from_exr(exr_file_path):
    f = OpenEXR.InputFile(exr_file_path)

    B = get_channel_from_exr(f, 'B')
    G = get_channel_from_exr(f, 'G')
    R = get_channel_from_exr(f, 'R')

    rgb = np.stack([R, G, B], axis=-1)
    return rgb


def load_apple(path, scale=1):
    images = []
    for img_file in sorted(glob.glob(os.path.join(path,"*[0-9]*.exr"))):
        img = data_from_exr(img_file)
        # img = cv.imread(img_file)[:,:,::-1].astype(np.float32) / 255.
        if scale!=1:
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
        images.append(img)
    images = np.stack(images, axis=0)

    mask_files = os.path.join(path, "mask.png")
    mask = cv.imread(mask_files, 0).astype(np.float32) / 255.

    if scale!=1:
        mask = cv.resize(mask, dim, interpolation=cv.INTER_LINEAR)

    light_dir_files = os.path.join(path, "light_directions_refined.txt")
    light_dir = parse_txt(light_dir_files)
    light_dir[..., 1:] = -light_dir[..., 1:]  # convert y-> -y   z->-z

    light_intensity_files = os.path.join(path, "light_intensities_refined.txt")
    light_intensity = parse_txt(light_intensity_files)

    gt_normal = np.zeros_like(images[0])

    out_dict = {'images': images, 'mask': mask, 'light_direction': light_dir, 'light_intensity': light_intensity, 'gt_normal': gt_normal}
    return out_dict
