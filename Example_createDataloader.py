from os.path import join
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import nibabel as nib
from torchvision.io import read_image
import matplotlib.pyplot as plt


# Creating a Custom Dataset for your files
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_labels.iloc[idx, 0]) # 1st column in .csv = filename
        image = nib.load(img_path).get_fdata()
        image = torch.tensor(image).permute([2,0,1])
        label = self.img_labels.iloc[idx, 1] # 2nd column in .csv = label (Normal=0, Abnormal=1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_csvdir = join('..', 'trainSet.csv')
train_imgdir = join('..', 'proc_data', 'TrainSet')
img_transform = nn.Sequential(
            T.Resize([224,224]),
            T.ConvertImageDtype(torch.float)
        )
training_data = CustomImageDataset(train_csvdir, train_imgdir, transform=img_transform)

# Preparing your data for training with DataLoaders
train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

# Iterate through the DataLoader
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Label: {label}")
plt.imshow(img[2,:,:], cmap="gray")
plt.show()