from data.base_dataset import BaseDataset, Rescale_fixed, Normalize_image
from data.image_folder import make_dataset, make_dataset_test

from augmentation import apply_augmentation

import os
import cv2
import json
import itertools
import collections
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image, ExifTags

import torch
import torchvision.transforms as transforms

import albumentations as A


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.input_dir = opt.input_dir
        self.label_dir = opt.label_dir

        custom_augs = apply_augmentation()
        transforms_list = []
        ###
        # Add Spatial-level transforms here
        # make sure tranforms are compatable with both input and label(mask/bbox/key points)
        ###
        transforms_list += [A.HorizontalFlip(p=0.5)]
        transforms_list += [A.VerticalFlip(p=0.01)]
        ###
        # apply_augmentation class from augmentation.py contains pixel-level transforms
        ###
        transforms_list += custom_augs(opt)
        self.transform_custom = A.Compose(transforms_list)

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(opt.mean, opt.std)]
        self.transform_input = transforms.Compose(transforms_list)

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        self.transform_label = transforms.Compose(transforms_list)

        self.input_list = sorted(make_dataset(self.input_dir))
        self.label_list = sorted(make_dataset(self.label_dir))

        self.dataset_size = len(self.input_list)

    def __getitem__(self, index):

        inp_path = self.input_list[index]
        lbl_path = self.label_list[index]

        ###
        # read input and label from inp_path and lbl_path resp.
        # apply above defined transforms as per need
        ###

        return input_tensor, label_tensor

    def __len__(self):
        return self.dataset_size

    def name(self):
        return "AlignedDataset"

    def orientation_fix(self, image):
        # Iphone and many other camera saves potrait mode photo into landscape,
        # fix for that
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break

            exif = image._getexif()
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
            return image
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            return image
