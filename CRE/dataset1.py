# coding=utf-8
import os
import pandas as pd
import math
import torch.utils.data as Data
import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.utils import save_image
from transforms import *
from skimage.transform import resize
import random
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold
import time
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
import shutil

import warnings
warnings.filterwarnings("ignore")

class Lung3D_ccii_patient_supcon(Data.Dataset):
    def __init__(self, train=False, inference=False, n_classes = 3):
        self.n_classes = n_classes
        self.img_size = 256
        self.train = train
        self.inference = inference
        self.datalist = []
        self.labelarraylist = []
        self.twocroptransform = TwoCropTransform(augment)

        '''cross validation'''
        self.root_dir = "/remote-home/my/2019-nCov/CC-CCII/data/3DCC-CCII-norm/"
        type_name = ['Normal','NCP','CP']
        all_pat_dir = [[],[],[]] 

        for i in range(len(type_name)):
            type = type_name[i]
            type_dir = self.root_dir + type + '/'
            for patient in sorted(os.listdir(type_dir)):
                all_pat_dir[i].append(patient)

        NCP = []
        CP = []
        Normal = []
        for i in range(len(type_name)):
            allpatdirs = all_pat_dir[i]
            if self.train:
                print(i, len(allpatdirs)) # all samples of a class

            KK = 0
            print("CV:",KK)
            kf = KFold(n_splits=5,shuffle=True,random_state=5)
            for k, (a,b) in enumerate(kf.split(range(len(allpatdirs)))):
                if k == KK:
                    train_index, test_index = a, b
                    if self.train:
                        print("before oversampling: ", len(train_index),len(test_index))
                        
            patient_index = train_index if train else test_index
            
            '''binary cls downsample''''
            # if train and i!= 1:
            #     np.random.seed(0)
            #     patient_index = np.random.choice(patient_index,int(0.6*len(patient_index)),replace=False)
            for index in patient_index:
                patient = allpatdirs[index]
                for scan in sorted(os.listdir(self.root_dir+type_name[i]+'/'+patient+'/')):
                    img = os.path.join(self.root_dir+type_name[i]+'/'+patient+'/', scan)
                    name = type_name[i]+'/'+patient+'/'+scan
                    self.datalist.append((img, i, name))

            print(len(self.datalist))

    def __getitem__(self, index):

        img, label, ID = self.datalist[index]
        img_array = np.load(img) 
        # print(index,ID,img_array.shape,label)

        img_array = rescale_z(img_array,64)

        if self.train:
            img_array = self.twocroptransform(img_array)
        else:
            img_array = normalize(img_array)
        
        '''binary cls'''
        # label = 1 if label == 1 else 0

        return img_array, label, ID       
    
    def __len__(self):
        return len(self.datalist)


def normalize(image):
    image = image / 255
    mean = 0.3529
    std = 0.2983
    image = 1.0 * (image - mean) / std
    return image

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [ torch.from_numpy(normalize(self.transform(x))).float(), torch.from_numpy(normalize(self.transform(x))).float()]

def rescale_z(images_zyx, target_depth, is_mask_image=False, verbose=False):
    # print("Resizing dim z")
    resize_x = 1.0
    resize_y = target_depth/images_zyx.shape[0]
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    return res

def augment(sample, ifhalfcrop=True,ifrandom_resized_crop=True, ifflip=False, ifrotate=False, ifcontrast=False,ifswap = False,filling_value=0):
    if ifhalfcrop:
        start = np.random.randint(0,sample.shape[0]-32)
        sample = sample[start:start+32]
    if ifrandom_resized_crop:
        rrc = RandomResizedCrop(size=256)
        i,j,w,h = rrc(sample)
        sample = sample[:,i:i+w,j:j+h]
        sample = rescale_gao(sample)
    if ifrotate:
        angle1 = np.random.rand()*45
        # size = np.array(sample.shape[2:4]).astype('float')
        rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
        sample = rotate(sample,angle1,axes=(1,2),reshape=False,cval=filling_value)
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            # coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
    if ifflip:
        flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1]])
        # coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
    if ifcontrast:
        contrast_low = 0.8
        contrast_up = 1.2
        brightness_low = -0.1
        brightness_up = 0.1
        c = np.random.uniform(contrast_low,contrast_up)
        b = np.random.uniform(brightness_low,brightness_up)
        sample = c * sample + b
        sample[sample>1] = 1
        sample[sample<0] = 0
    return sample

def rescale_gao(images_zyx, is_mask_image=False):
    res = images_zyx
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR

    if res.shape[0] > 512:
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(256, 256), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(256, 256), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = np.vstack([res1, res2])
        # res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res.transpose(1, 2, 0), dsize=(256,256), interpolation=interpolation)
        res = res.transpose(2, 0, 1)
    # print("Shape after: ", res.shape)
    return res

class RandomResizedCrop(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size

        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (tuple): range of scale of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        _, width, height = img.shape
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(*scale).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return i, j, h, w
        # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


if __name__ == "__main__":
    dst = Lung3D_ccii_patient_supcon(train=True,n_classes=3)
    exit()
    for i in range(len(dst)):
        img_array, label, ID = dst.__getitem__(i) 
        break
    exit()  
    from functions import get_mean_and_std
    mean, std = get_mean_and_std(dst)
    print(mean, std)
    exit()


