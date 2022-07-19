# borrows from https://github.com/guanyingc/SDPS-Net
import os
import numpy as np
import re

from imageio import imread

import torch
import torch.utils.data as data

import pms_transforms
np.random.seed(0)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def readList(list_path,ignore_head=False, sort=True):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists


class UPS_Synth_Dataset(data.Dataset):
    def __init__(self, root, split='train'):
        self.root  = os.path.join(root)
        self.split = split
        self.images_per_object = 1
        self.shape_list = readList(os.path.join(self.root, split + "_mtrl.txt"))

    def _getInputPath(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
        img_dir     = os.path.join(self.root, 'Images', self.shape_list[index])
        img_list    = readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        select_idx = np.random.permutation(data.shape[0])[:self.images_per_object]
        idxs = ['%03d' % (idx) for idx in select_idx]
        data = data[select_idx, :]
        imgs = [os.path.join(img_dir, img) for img in data[:, 0]]
        dirs = data[:, 1:4].astype(np.float32)
        return normal_path, imgs, dirs

    def __getitem__(self, index):
        normal_path, img_list, dirs = self._getInputPath(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        imgs   =  []
        for i in img_list:
            img = imread(i).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = 128, 128
        if not (crop_h == h):
            sc_h = np.random.randint(crop_h, h)
            sc_w = np.random.randint(crop_w, w)
            img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])

        img = img * np.random.uniform(1, 3)

        if True:
            ints = pms_transforms.getIntensity(len(imgs))
            img  = np.dot(img, np.diag(ints.reshape(-1)))
        else:
            ints = np.ones(c)

        img = pms_transforms.randomNoiseAug(img, 0.05)

        mask   = pms_transforms.normalToMask(normal)
        normal = normal * mask.repeat(3, 2)
        norm   = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10) # Rescale normal to unit length

        normal[..., 1:] = -normal[..., 1:]  # convert y-> -y   z->-z

        item = {'normal': normal, 'img': img, 'mask': mask}
        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        dirs[..., 1:] = -dirs[..., 1:]  # convert y-> -y   z->-z

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        return item

    def __len__(self):
        return len(self.shape_list)


def customDataloader(data_dir, data_dir2, batch=32, val_batch=32, workers=8, cuda=False):
    print("=> fetching img pairs in %s" % (data_dir))

    train_set = UPS_Synth_Dataset(data_dir, 'train')
    val_set   = UPS_Synth_Dataset(data_dir, 'val')

    print('****** Using cocnat data ******')
    print("=> fetching img pairs in '{}'".format(data_dir2))
    train_set2 = UPS_Synth_Dataset(data_dir2, 'train')
    val_set2   = UPS_Synth_Dataset(data_dir2, 'val')

    train_set  = torch.utils.data.ConcatDataset([train_set, train_set2])
    val_set    = torch.utils.data.ConcatDataset([val_set,   val_set2])

    print('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    print('\t Train Batch: %d, Val Batch: %d' % (batch, val_batch))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch,
        num_workers=workers, pin_memory=cuda, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=val_batch,
        num_workers=workers, pin_memory=cuda, shuffle=False)
    return train_loader, test_loader
