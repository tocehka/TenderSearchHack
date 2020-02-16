import os
import json
import datetime

import numpy as np

import cv2
import torch

import torch.utils.data as data


class Amazon(data.Dataset):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.images_list = self._get_images_list(self.folder)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self._get_image(self.images_list[idx])

        if self.transforms:
            image, _, _ = self.transforms(image)
            
        return image
 
    def _get_image(self, file_path):
        image = cv2.imread(os.path.join(self.folder, file_path), cv2.IMREAD_COLOR)
        return image

    def _get_images_list(self, folder):
        images_list = os.listdir(folder)
        return images_list