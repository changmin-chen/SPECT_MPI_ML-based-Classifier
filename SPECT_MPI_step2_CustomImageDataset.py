from os.path import join
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob2 as gb


# Creating a Custom Dataset for your files
# appended channel: = ch1 - ch0
# output format of images: [C, H, W, D]
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, dataroot, transform=None):
        self.labels = pd.read_csv(annotations_file)
        images_path = gb.glob(join(dataroot, '*.nii'))
        self.images = [nib.load(path).get_fdata() for path in images_path] # save all data in the RAM
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


# Computing mean & std for each channel, for Standardization of the dataset.
# if your RAM can load all samples at onece (like dataset for this project), the code should be much shorter
# however, this should not always be a case.
def dataset_statistics(dataset):
    batch_size = 16
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.zeros(3, dtype=torch.float32)
    nsample = 0
    for images, _ in loader:
        (B, C, H, W, D) = images.shape
        nsample += B
        images = images.view(B, C, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= nsample
    std /= nsample

    return mean, std


# Display mean & std for each channel, operated across all image processing versions
def main():
    
    nver = 3 # total number of image processing versions
    for ver in range(nver):
        foldername = 'proc_data_ver'+ str(ver)
        train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', foldername, 'TrainSet'))
        mean, std = dataset_statistics(train_dataset)

        # These values should be applied for Standardization of the train-dataset before training
        print(f'Image processing ver {ver} statistics:\n mean = {mean}, std = {std}')


if __name__=='__main__':
    main()