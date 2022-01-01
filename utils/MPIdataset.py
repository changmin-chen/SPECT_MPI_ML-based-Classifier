from os.path import join
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob2 as gb


# Create a Custom Dataset for your files
# appended channel: = ch1 - ch0
# output format of images: [C, H, W, D]
class MPIdataset(Dataset):
    def __init__(self, annotations_file, dataroot, transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.imagepaths = gb.glob(join(dataroot, '*.nii'))
        self.images = [nib.load(path).get_fdata() for path in self.imagepaths] # save all data in the RAM
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 2nd column in csv contains labels (Normal=0, Abnormal=1)
        label = torch.tensor(self.labels.iloc[idx, 1])
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute([3,0,1,2])
        image_cat = torch.unsqueeze(image[1] - image[0], dim=0) 
        image = torch.cat((image, image_cat), dim=0)

        if self.transform:
            image = self.transform(image)

        return image, label