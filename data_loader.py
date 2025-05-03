from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
import cv2
from glob import glob
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F


#### --------------------- DIS dataloader ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + '*' + datasets[i]["im_ext"])

        # img_name_dict[im_dirs[i][0]] = tmp_im_list
        print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

        if (datasets[i]["gt_dir"] == ""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] +
                           datasets[i]["gt_ext"] for x in tmp_im_list]

            # lbl_name_dict[im_dirs[i][0]] = tmp_gt_list
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))

        if flag == "train":  ## combine multiple training sets into one dataset
            if len(name_im_gt_list) == 0:
                name_im_gt_list.append({"dataset_name": datasets[i]["name"],
                                        "im_path": tmp_im_list,
                                        "gt_path": tmp_gt_list,
                                        "im_ext": datasets[i]["im_ext"],
                                        "gt_ext": datasets[i]["gt_ext"],
                                        "cache_dir": datasets[i]["cache_dir"]})
            else:
                name_im_gt_list[0]["dataset_name"] = name_im_gt_list[0]["dataset_name"] + "_" + datasets[i][
                    "name"]
                name_im_gt_list[0]["im_path"] = name_im_gt_list[0]["im_path"] + tmp_im_list
                name_im_gt_list[0]["gt_path"] = name_im_gt_list[0]["gt_path"] + tmp_gt_list
                if datasets[i]["im_ext"] != ".jpg" or datasets[i]["gt_ext"] != ".png":
                    print(
                        "Error: Please make sure all you images and ground truth masks are in jpg and png format respectively !!!")
                    exit()
                name_im_gt_list[0]["im_ext"] = ".jpg"
                name_im_gt_list[0]["gt_ext"] = ".png"
                name_im_gt_list[0]["cache_dir"] = os.sep.join(
                    datasets[i]["cache_dir"].split(os.sep)[0:-1]) + os.sep + name_im_gt_list[0]["dataset_name"]
        else:  ## keep different validation or inference datasets as separate ones
            name_im_gt_list.append({"dataset_name": datasets[i]["name"],
                                    "im_path": tmp_im_list,
                                    "gt_path": tmp_gt_list,
                                    "im_ext": datasets[i]["im_ext"],
                                    "gt_ext": datasets[i]["gt_ext"],
                                    "cache_dir": datasets[i]["cache_dir"]})

    return name_im_gt_list


def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, shuffle=False):
    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    for i in range(0, len(name_im_gt_list)):
        gos_dataset = GOSDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms))
        gos_dataloaders.append(
            DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


class GOSRandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSRandomVFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[0])
            label = torch.flip(label, dims=[0])

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSResize(object):
    def __init__(self, size=None):
        if size is None:
            size = [320, 320]
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # import time
        # start = time.time()

        image = torch.squeeze(F.upsample(torch.unsqueeze(image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.upsample(torch.unsqueeze(label, 0), self.size, mode='bilinear'), dim=0)

        # print("time for resize: ", time.time()-start)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSRandomCrop(object):
    def __init__(self, size=None):
        if size is None:
            size = [288, 288]
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class GOSDataset(Dataset):

    def __init__(self, name_im_gt_list, transform=None):
        self.transform = transform
        self.dataset = {}

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per image
        im_name_list = []  # image name
        im_path_list = []  # im path
        gt_path_list = []  # gt path
        im_ext_list = []  # im ext
        gt_ext_list = []  # gt ext
        for i in range(0, len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in
                                 name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_shp"] = []
        self.dataset["gt_shp"] = []
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]

        im = cv2.imread(im_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Resize both image and ground truth to have the same spatial dimensions
        im, gt = self.ensure_consistent_size(im, gt)

        im_shp = im.shape[:2]

        im = torch.tensor(im, dtype=torch.float32) / 255.0
        gt = torch.tensor(gt, dtype=torch.float32) / 255.0

        sample = {
            "imidx": torch.tensor(idx),
            "image": im.permute(2, 0, 1),
            "label": gt.unsqueeze(0),
            "shape": torch.tensor(im_shp),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def ensure_consistent_size(self, im, gt):
        # Check if spatial dimensions match
        if im.shape[:2] != gt.shape[:2]:
            # Resize both image and ground truth to have the same spatial dimensions
            im = resize(im, gt.shape[:2], mode='constant', preserve_range=True)

        return im, gt