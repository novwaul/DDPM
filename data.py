from os import listdir
from os.path import join
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np


class CelebADataset(Dataset):
    def __init__(self):
        self.path = "/home/kaist/inje/sr/data/celeba_hq_256"
        self.names = sorted([name for name in listdir(self.path)])
        self.tensor = transforms.ToTensor()
        self.random_flip = transforms.RandomHorizontalFlip()
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(join(self.path, self.names[idx]))
        img = self.tensor(img)
        img = self.random_flip(img)

        return img