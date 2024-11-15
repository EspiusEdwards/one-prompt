
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + "_Data")
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index] + ".jpg"
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.name_list[index] + "_Segmentation.png"
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict
            # 'image_path': img_path,       # Added line
            # 'label_path': msk_path        # Added line
        }
### ISICS loader
from torch.utils.data import DataLoader

def get_isic_loader(args):
    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # Add any other image transformations you need
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # Ensure mask is binary (0 or 1)
        transforms.Lambda(lambda x: (x > 0).float()),
    ])

    # Create training and validation datasets
    train_dataset = ISIC2016(
        args=args,
        data_path=args.data_path,
        transform=transform,
        transform_msk=transform_mask,
        mode='Training',
        prompt='click'
    )

    val_dataset = ISIC2016(
        args=args,
        data_path=args.data_path,
        transform=transform,
        transform_msk=transform_mask,
        mode='Test',  # or 'Validation' if you have a validation set
        prompt='click'
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
