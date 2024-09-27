import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image

class CutCov_Fix(object):
    def __init__(self, Probability=1.0, Ratio=0.2):
        self.P = Probability
        self.R = Ratio

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.P:
            return img

        H = img.size()[1]
        W = img.size()[2]

        cut_area = H * W * self.R
        cut_d = int(math.sqrt(cut_area))

        y1 = np.random.randint(0, H - cut_d + 1)
        x1 = np.random.randint(0, W - cut_d + 1)
        y2 = np.random.randint(0, H - cut_d + 1)
        x2 = np.random.randint(0, W - cut_d + 1)

        img1 = img.clone()
        img[:, y1:y1 + cut_d, x1:x1 + cut_d] = img1[:, y2:y2 + cut_d, x2:x2 + cut_d]

        return img

class CutRot(object):
    def __init__(self, Probability=1.0, Ratio=0.2):
        self.P = Probability
        self.R = Ratio

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.P:
            return img

        H = img.size()[1]
        W = img.size()[2]

        cut_area = H * W * self.R
        cut_d = int(math.sqrt(cut_area))

        ## determine coordinate point of patch
        y = np.random.randint(0, H - cut_d + 1)
        x = np.random.randint(0, W - cut_d + 1)

        k = np.random.choice([0, 1, 2, 3])
        img1 = img.clone()
        img[0, y:y + cut_d, x:x + cut_d] = torch.rot90(img1[0, y:y + cut_d, x:x + cut_d], k, dims=[0,1])
        img[1, y:y + cut_d, x:x + cut_d] = torch.rot90(img1[1, y:y + cut_d, x:x + cut_d], k, dims=[0,1])
        img[2, y:y + cut_d, x:x + cut_d] = torch.rot90(img1[2, y:y + cut_d, x:x + cut_d], k, dims=[0,1])

        return img


def get_transform(is_training=None,transform_type=None,pre=None):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    if transform_type and pre:
        raise Exception('transform_type and pre cannot be specified as True at the same time')

    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,std=std)
                                    ])

    if is_training:

        if transform_type == 0:
            size_transform = transforms.RandomResizedCrop(84)
        elif transform_type == 1:
            size_transform = transforms.RandomCrop(84,padding=8)
        else:
            raise Exception('transform_type must be specified during training!')
        
        train_transform = transforms.Compose([size_transform,
                                            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                            transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              CutRot(1, 0.2),
                                              CutCov_Fix(1, 0.2),
                                              transforms.Normalize(mean=mean, std=std)
                                            ])
        return train_transform
    
    elif pre:
        return normalize
    
    else:
        
        if transform_type == 0:
            size_transform = transforms.Compose([transforms.Resize(92),
                                                transforms.CenterCrop(84)])
        elif transform_type == 1:
            size_transform = transforms.Compose([transforms.Resize([92,92]),
                                                transforms.CenterCrop(84)])
        elif transform_type == 2:
            # for tiered-imagenet and (tiered) meta-inat where val/test images are already 84x84
            return normalize

        else:
            raise Exception('transform_type must be specified during inference if not using pre!')
        
        eval_transform = transforms.Compose([size_transform,normalize])
        return eval_transform
