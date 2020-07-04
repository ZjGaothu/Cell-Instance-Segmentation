# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import imageio
import os.path as osp
import os

def load_images(file_names):
    images = []
    for file_name in file_names:
        img = io.imread(file_name)
        images.append(img)
    return images

def preprocess(sample,newsize,istraining,cropsize):
    imidx, image, label = sample['imidx'], sample['image'],sample['label']
#     if random.random() >= 0.5:
#         image = image[::-1]
#         label = label[::-1]
    h, w = image.shape[:2]
    new_h, new_w = newsize,newsize*w/h
    new_h, new_w = int(new_h), int(new_w)
    image = transform.resize(image,(newsize,newsize),mode='constant')
    label = transform.resize(label,(newsize,newsize),mode='constant', order=0, preserve_range=True)
    if(istraining == 'train'):
        init_x = math.floor(np.random.rand()*(newsize-cropsize))
        init_y = math.floor(np.random.rand()*(newsize-cropsize))
        image = image[init_x:init_x+cropsize, init_y:init_y+cropsize]
        label = label[init_x:init_x+cropsize, init_y:init_y+cropsize]
    tmpLbl = np.zeros(label.shape)
    
    if(np.max(label)<1e-6):
        label = label
    else:
        label = label/np.max(label)
    tmpImg = np.zeros((image.shape[0],image.shape[1],3))
    image = image/np.max(image)
    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
    tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
    tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
#     tmpLbl[:,:,0] = label[:,:,0]
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpLbl = label.transpose((2, 0, 1))
    return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}
    
class Celldataset(Dataset):
    def __init__(self,img_list,mask_list,istraining,transform=None):
        super().__init__()
#         self.image_list = [osp.join(img_path, image) for image in os.listdir(img_path)]
#         self.mask_list = [osp.join(mask_path, image) for image in os.listdir(mask_path)]
#         self.images = load_images(self.image_list)
#         self.masks = load_images(self.mask_list)
        self.transform = transform
        self.istrain = istraining
        self.image_list=img_list
        self.mask_list=mask_list
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image = io.imread(self.image_list[idx]).astype(np.float)
        mask = io.imread(self.mask_list[idx]).astype(np.float)
        imageidx = np.array([idx])
        image = image[:,:,np.newaxis]
        mask = mask[:,:,np.newaxis]
        
        value = {'imidx':imageidx, 'image':image, 'label':mask}
#         if self.istrain == 'train'
        value = preprocess(value,320,self.istrain,288)

        return value