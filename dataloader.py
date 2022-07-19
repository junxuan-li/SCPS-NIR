import torch
from torch.utils.data import Dataset
import numpy as np
import load_diligent
import cv2 as cv


class Data_Loader(Dataset):
    def __init__(self, data_dict, gray_scale=False, data_len=1, mode='training', shadow_threshold=0.0):
        self.light_intensity = torch.tensor(data_dict['light_intensity'], dtype=torch.float32)
        self.images = torch.tensor(data_dict['images'], dtype=torch.float32)  # (num_images, height, width, channel)

        if gray_scale:
            self.images = self.images.mean(dim=-1, keepdim=True)  # (num_images, height, width, 1)
            self.light_intensity = self.light_intensity.mean(dim=-1, keepdim=True)
        self.images_max = torch.tensor(1.0, dtype=torch.float32)  # self.images.max()  #  torch.tensor(0.5, dtype=torch.float32)

        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        self.light_direction = torch.tensor(data_dict['light_direction'], dtype=torch.float32)

        self.gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)
        self.pre_contour_normal = self.compute_contour_normal(data_dict['mask'])

        self.num_images = self.images.size(0)
        self.height = self.images.size(1)
        self.width = self.images.size(2)
        masks = self.mask[None,...].repeat((self.num_images,1,1))  # (num_images, height, width)

        self.valid_idx = torch.where(masks > 0.5)
        temp_idx = torch.where(self.mask > 0.5)

        self.valid_rgb = self.images[self.valid_idx]

        self.valid_input_iwih = torch.stack([temp_idx[1] / self.width, temp_idx[0] / self.height], dim=-1)
        self.valid_input_iwih_max, _ = self.valid_input_iwih.max(dim=0)
        self.valid_input_iwih_min, _ = self.valid_input_iwih.min(dim=0)
        self.mean_valid_iwih = self.valid_input_iwih.mean(dim=0, keepdim=True)

        self.valid_input_iwih = self.valid_input_iwih - self.mean_valid_iwih

        self.num_valid_rays = int(self.valid_input_iwih.size(0))
        self.valid_light_direction = torch.repeat_interleave(self.light_direction, self.num_valid_rays, dim=0)

        self.data_len = min(data_len, self.num_images)
        self.mode = mode

        images_mean = torch.mean(self.images, dim=0)  # (height, width, channel)
        images_var = torch.var(self.images, dim=0)  # (height, width, channel)
        temp_mean_var = torch.cat([images_mean, images_var], dim=-1)  # (height, width, channel*2)

        self.valid_images_meanvar = temp_mean_var[temp_idx]

        self.valid_light_direction = self.valid_light_direction.view(self.num_images, -1, 3)
        self.valid_rgb = self.valid_rgb.view(self.num_images, -1, 1 if gray_scale else 3)
        self.valid_gt_normal = self.gt_normal[temp_idx]

        self.valid_shadow = None
        self.update_valid_shadow_map(thres=shadow_threshold)
        self.get_contour_idx()

    def __len__(self):
        if self.mode == 'testing':
            return self.data_len
        else:
            raise NotImplementedError('Dataloader mode unknown')

    def __getitem__(self, idx):
        if self.mode == 'training':
            return self.get_all_rays()
        if self.mode == 'testing':
            return self.get_testing_rays(idx)

    def get_all_rays(self):
        idx = torch.randperm(self.num_images)

        input_xy = self.valid_input_iwih
        input_light_direction = self.valid_light_direction[idx]
        rgb = self.valid_rgb[idx]
        normal = self.valid_gt_normal[idx]

        light_intensity = self.light_intensity[idx]

        sample = {'input_xy': input_xy, 'input_light_direction': input_light_direction, 'light_intensity': light_intensity, 'rgb': rgb, 'normal': normal}

        return sample

    def get_testing_rays(self, ith):
        input_xy = self.valid_input_iwih
        input_light_direction = self.valid_light_direction[ith]
        rgb = self.valid_rgb[ith]
        normal = self.valid_gt_normal

        light_intensity = self.light_intensity[ith]

        mean_var = self.valid_images_meanvar

        sample = {'input_xy': input_xy,
                  'input_light_direction': input_light_direction,
                  'light_intensity': light_intensity,
                  'rgb': rgb,
                  'normal': normal,
                  'mean_var': mean_var,
                  'item_idx': ith}

        sample['shadow_mask'] = self.valid_shadow[ith]
        sample['contour'] = self.contour
        sample['contour_normal'] = self.contour_normal

        dx = 1 / self.mask.size(1)
        dy = 1 / self.mask.size(1)
        px = torch.zeros_like(input_light_direction)
        px[:, 0] = 2 * dx
        py = torch.zeros_like(input_light_direction)
        py[:, 1] = 2 * dy
        sample['px'] = px
        sample['py'] = py

        return sample

    def get_mask(self):
        return self.mask

    def get_mean_xy(self):
        return self.mean_valid_iwih

    def get_bounding_box(self):
        return self.valid_input_iwih_max, self.valid_input_iwih_min

    def get_bounding_box_int(self):
        mask = self.mask.numpy()
        valididx = np.where(mask > 0.5)
        xmin = valididx[0].min()
        xmax = valididx[0].max()
        ymin = valididx[1].min()
        ymax = valididx[1].max()

        xmin = max(0, xmin - 1)
        xmax = min(xmax + 2, mask.shape[0])
        ymin = max(0, ymin - 1)
        ymax = min(ymax + 2, mask.shape[1])
        return xmin, xmax, ymin, ymax

    def get_all_light_direction(self):
        return self.light_direction

    def get_all_light_intensity(self):
        return self.light_intensity

    def get_all_light_encoding(self):
        return self.ld_encoding

    def get_all_masked_images(self):
        idx = torch.where(self.mask > 0.5)
        x_max, x_min = max(idx[0]), min(idx[0])
        y_max, y_min = max(idx[1]), min(idx[1])

        x_max, x_min = min(x_max+15, self.images.shape[1]), max(x_min-15, 0)
        y_max, y_min = min(y_max+15, self.images.shape[2]), max(y_min-15, 0)

        out_images = self.images[:, x_min:x_max, y_min:y_max, :].permute([0,3,1,2])
        out_masks = self.mask[x_min:x_max, y_min:y_max][None, None, ...].repeat(out_images.size(0),1,1,1)
        out = torch.cat([out_images, out_masks], dim=1)
        return out  # (num_image, 4, height, width)

    def update_valid_shadow_map(self, thres):
        if self.valid_rgb.size(-1) == 3:
            temp_rgb = self.valid_rgb.mean(dim=-1)  # (num_image, num_mask_point)
        else:
            temp_rgb = self.valid_rgb
        temp_rgb_topk_mean = torch.topk(temp_rgb, k=int(len(temp_rgb)*0.9), dim=0, largest=False)[0].mean(dim=0, keepdim=True)

        idxp = torch.where(thres*temp_rgb_topk_mean <= temp_rgb)

        self.valid_shadow = torch.zeros_like(temp_rgb)
        self.valid_shadow[idxp] = 1
        return

    def update_valid_shadow_map_from_pth(self, path, thres=0.01):
        temp_render = torch.tensor(np.load(path), dtype=torch.float32)

        if self.valid_rgb.size(-1) == 3:
            temp_rgb = self.valid_rgb.mean(dim=-1)  # (num_image, num_mask_point)
        else:
            temp_rgb = self.valid_rgb
        temp_rgb_topk_mean = torch.topk(temp_rgb, k=len(temp_rgb)-11, dim=0, largest=False)[0].mean(dim=0, keepdim=True)

        idxp = torch.where(thres*temp_rgb_topk_mean <= temp_rgb)

        temp_thres = torch.zeros_like(temp_rgb)
        temp_thres[idxp] = 1

        self.valid_shadow = temp_thres * temp_render
        return

    def get_contour_idx(self):
        mask_x1, mask_x2, mask_y1, mask_y2 = self.mask.clone(), self.mask.clone(), self.mask.clone(), self.mask.clone()
        mask_x1[:-1, :] = self.mask[1:, :]
        mask_x2[1:, :] = self.mask[:-1, :]
        mask_y1[:, :-1] = self.mask[:, 1:]
        mask_y2[:, 1:] = self.mask[:, :-1]
        mask_1 = mask_x1 * mask_x2 * mask_y1 * mask_y2
        idxp_contour = torch.where((mask_1 < 0.5) & (self.mask > 0.5))

        contour_map = torch.zeros_like(self.mask)
        contour_map[idxp_contour] = 1

        self.contour = contour_map[torch.where(self.mask>0.5)]
        self.contour_normal = self.contour[:,None] * self.pre_contour_normal[torch.where(self.mask>0.5)]
        return idxp_contour

    def get_guess_light_intensities(self):
        _, guess_li = self.valid_rgb.mean(dim=-1).max(dim=-1)
        guess_li = guess_li / guess_li[0]

        return guess_li[:, None].repeat(1,3)

    @staticmethod
    def compute_contour_normal(_mask):
        blur = cv.GaussianBlur(_mask, (11, 11), 0)
        n_x = -cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        n_y = -cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        n = np.sqrt(n_x**2 + n_y**2) + 1e-5
        contour_normal = np.zeros((_mask.shape[0], _mask.shape[1], 3), np.float32)
        contour_normal[:, :, 0] = n_x / n
        contour_normal[:, :, 0] = n_x / n
        return torch.tensor(contour_normal, dtype=torch.float32)

    @staticmethod
    def get_unitsphere_normal():
        data_dict = load_diligent.load_unitsphere()
        mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)
        valid_idx = torch.where(mask > 0.5)
        invalid_idx = torch.where(mask < 0.5)
        valid_gt_normal = gt_normal[valid_idx]
        return valid_gt_normal, valid_idx, invalid_idx

    @staticmethod
    def get_unitsphere_bounding_box_int():
        data_dict = load_diligent.load_unitsphere()
        mask = data_dict['mask']

        valididx = np.where(mask > 0.5)
        xmin = valididx[0].min()
        xmax = valididx[0].max()
        ymin = valididx[1].min()
        ymax = valididx[1].max()

        xmin = max(0, xmin - 1)
        xmax = min(xmax + 2, mask.shape[0])
        ymin = max(0, ymin - 1)
        ymax = min(ymax + 2, mask.shape[1])
        return xmin, xmax, ymin, ymax
