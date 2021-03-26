# coding=utf-8
import os
import pandas as pd
import math
import torch.utils.data as Data
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import transforms
from skimage.transform import resize
import random
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold
import time
from functions import get_mean_and_std

import warnings
warnings.filterwarnings("ignore")


class CCII_distmap(Data.Dataset):
    def __init__(self, train=True):
        self.img_size = 256
        self.train = train
        self.datalist = []
        self.distlist = []

        self.root_dir = "/remote-home/my/2019-nCov/CC-CCII/data/ct_lesion_seg/my/"
        
        img_dir = self.root_dir + 'image2/'
        label_dir = self.root_dir + 'dist2/'

        allimgdirs = sorted(os.listdir(img_dir))
        if self.train:
            print(len(allimgdirs)) # all samples of a class
        
        KK = 0
        print("CV:",KK)
        kf = KFold(n_splits=5,shuffle=True,random_state=5)
        for j, (a,b) in enumerate(kf.split(range(len(allimgdirs)))):
            if j == KK:
                train_index, test_index = a, b
                if self.train:
                    print("before oversampling: ", len(train_index),len(test_index))

        data_index = train_index if train else test_index
        for index in data_index:
            item = allimgdirs[index]
            patient = os.path.join(img_dir, item)
            patient_dist = os.path.join(label_dir, item.replace('jpg','mat'))
            self.datalist.append(patient)
            self.distlist.append(patient_dist)
              
        print(len(self.datalist))             

    def __getitem__(self, index):

        img = self.datalist[index]
        ID = img.split('/')[-1].split('.')[0]
        dist = self.distlist[index]

        img_array = cv2.imread(img)

        dist_dict = loadmat(dist)
        dist_array = dist_dict['dists']
        
        left, right, front, behind = getbbox_2d(dist_array)
        s = (right-left)*(behind-front)
        eps = 1e-10
        dist_array = dist_array/(s+eps)
        dist_array = dist_array/(dist_array.max()+eps)

        if self.train:
            img_array, dist_array = augment_mask(img_array,dist_array)
        else:
            scale = Scale(size=256)
            img_array, dist_array = scale(img_array,dist_array)

        normalize = T.Normalize(mean=[0.5267, 0.5267, 0.5267], std=[0.3543, 0.3543, 0.3543])
        transformer = T.Compose([
            T.ToTensor(),
            normalize
        ])
        img_array = transformer(img_array)

        dist_array = np.clip(dist_array, 0, 1)
        dist_array[dist_array == 0] = 2
        dist_array[dist_array <= 0.15] = 3
        dist_array[dist_array <= 0.3] = 4
        dist_array[dist_array <= 0.45] = 5
        dist_array[dist_array <= 0.6] = 6
        dist_array[dist_array <= 1] = 7
        dist_array -= 2   
        # print(ID,np.unique(dist_array))

        return img_array, dist_array, ID     
    
    def __len__(self):
        return len(self.datalist)


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, dist):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.7, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[0] and h <= img.shape[1]:
                x1 = random.randint(0, img.shape[0] - w)
                y1 = random.randint(0, img.shape[1] - h)

                # img = img.crop((x1, y1, x1 + w, y1 + h))
                img = img[x1 : x1 + w, y1 : y1 + h]
                dist = dist[x1 : x1 + w, y1 : y1 + h]
                assert((img.shape[0],img.shape[1]) == (w, h))

                img = cv2.resize(img, dsize=(self.size,self.size), interpolation=self.interpolation)
                # dist = cv2.resize(dist, dsize=(self.size,self.size), interpolation=self.interpolation)
                dist = cv2.resize(dist, dsize=(self.size,self.size), interpolation=Image.NEAREST)
                return img,dist

                # return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        # crop = CenterCrop(self.size)
        # return crop(scale(img))
        return scale(img,dist)

class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        import collections
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 3)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img, mask):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h, d = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                # return img.resize((ow, oh), self.interpolation)
                return cv2.resize(img, dsize=(ow,oh), interpolation=self.interpolation),\
                       cv2.resize(mask, dsize=(ow,oh), interpolation=Image.NEAREST)

            else:
                oh = self.size
                ow = int(self.size * w / h)
                # return img.resize((ow, oh), self.interpolation)
                return cv2.resize(img, dsize=(ow,oh), interpolation=self.interpolation),\
                       cv2.resize(mask, dsize=(ow,oh), interpolation=Image.NEAREST)
        else:
            # return img.resize(self.size, self.interpolation)
            return cv2.resize(img, dsize=self.size, interpolation=self.interpolation),\
                   cv2.resize(mask, dsize=(ow,oh), interpolation=Image.NEAREST)

class RandomFlip(object):

    def __init__(self, lr_prob: float=0, ud_prob: float=0):
        self.lr = lr_prob
        self.ud = ud_prob

    def __call__(self, sample, dist):
        flip_count = None
        if self.lr and np.random.uniform() < self.lr:
            flip_count = 1
        if self.ud and np.random.uniform() < self.ud:
            flip_count = 0 if flip_count is None else -1
        if flip_count is not None:
            sample = cv2.flip(sample, flip_count)
            dist = cv2.flip(dist, flip_count)
        return sample, dist

class RandomContrastAndBrightness(object):

    def __init__(self, contrast: tuple or float=1.0, brightness: tuple or float=0.0):
        if isinstance(contrast, tuple):
            self.contrast_low = contrast[0]
            self.contrast_up = contrast[1]
        if isinstance(brightness, tuple):
            self.brightness_low = brightness[0]
            self.brightness_up = brightness[1]
        if isinstance(contrast, float):
            self.contrast_low = self.contrast_up = contrast
        if isinstance(brightness, float):
            self.brightness_low = self.brightness_up = brightness

    def __call__(self, sample):
        c = np.random.uniform(self.contrast_low, self.contrast_up)
        b = np.random.uniform(self.brightness_low, self.brightness_up)
        sample = c * sample + b
        sample[sample > 1.0] = 1.
        return sample

def augment_mask(sample, mask, ifrandom_resized_crop=True, ifflip=True, ifcontrast=False):
    if ifrandom_resized_crop:
        rrc = RandomSizedCrop(size=256)
        sample, mask = rrc(sample, mask)
    if ifflip:
        flip = RandomFlip(lr_prob=0.5, ud_prob=0.5)
        sample, mask = flip(sample, mask)     
    if ifcontrast:
        rcb = RandomContrastAndBrightness(contrast=(0.9, 1.0),brightness=(0.9,1.0))
        sample = rcb(sample)
    return sample, mask

def getbbox_2d(img):
    w, h = img.shape
    margin = 5
    left, right, front, behind = 0,0,0,0
    for i in range(w):
        if img[i,:].sum()>0:
            left = max(0, i - margin)
            break
    for i in range(w-1,-1,-1):
        if img[i,:].sum()>0:
            right = min(w-1, i + margin)
            break
    for i in range(h):
        if img[:,i].sum()>0:
            front = max(0, i - margin)
            break
    for i in range(h-1,-1,-1):
        if img[:,i].sum()>0:
            behind = min(h-1, i + margin)
            break   
    return left, right, front, behind

if __name__ == "__main__":
    dst = CCII_distmap(train=True)
    # mean, std = get_mean_and_std(dst)
    # print(mean,std)

    for i in range(dst.__len__()):
        img_array, label, ID = dst.__getitem__(i)
        print(img_array.shape)
        print('------------------------------------')
