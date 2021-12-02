from os.path import join
import pandas as pd
import nibabel as nib
import torch
from torch._C import dtype
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob2 as gb


# Creating a Custom Dataset for your files
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, dataroot):
        self.labels = pd.read_csv(annotations_file)
        images_path = gb.glob(join(dataroot, '*.nii'))
        self.images = [nib.load(path).get_fdata() for path in images_path] # save all data in the RAM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels.iloc[idx, 1]) # 2nd column in csv = label (Normal=0, Abnormal=1)
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute([3,0,1,2]) # permuted tensor would be [Channel x Height x Width x Depth]
        image_cat = torch.unsqueeze(image[1] - image[0], dim=0) # because inputs sholud have 3 channel
        image = torch.cat((image, image_cat), dim=0)

        return image, label
     


# Example: usage of CustomImageDataset
if __name__ == '__main__':

    # Dataset
    train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', 'proc_data_ver0', 'TrainSet'))
    test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', 'proc_data_ver0', 'TestSet'))

    # WeightedRandomSampler for unbalance data
    weights = 1. /torch.tensor([35, 130], dtype=torch.float32)
    train_target = torch.tensor(train_dataset.labels.iloc[:, 1], dtype=torch.long)
    sample_weights = weights[train_target]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    # Data loader (input pipeline)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # Show up output tensor.Size
    for _, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        print(labels)


# Using "CustomImageDataset" in SPECT_MPI_step3_dataLoader:
#
#   from SPECT_MPI_step3_dataLoader import CustomImageDataset