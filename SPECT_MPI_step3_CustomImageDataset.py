from os.path import join
import pandas as pd
import nibabel as nib
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
        img_path = join(self.img_dir, self.img_labels.iloc[idx, 0]) # 1st column in csv = filename
        image = nib.load(img_path).get_fdata() # readout nii file as numpyArray using nib package
        image = torch.tensor(image).permute([3,0,1,2]) # permuted tensor would be [Channel x Height x Width x Depth]
        label = self.img_labels.iloc[idx, 1] # 2nd column in csv = label (Normal=0, Abnormal=1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Example: usage of CustomImageDataset
if __name__ == '__main__':

    # Define transform and/or target_transform (this is optional)
    img_transform = torch.nn.Sequential(
        T.ConvertImageDtype(torch.float),
    )

    # Dataset
    train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', 'proc_data', 'TrainSet'), transform=img_transform)
    test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', 'proc_data', 'TestSet'), transform=img_transform)

    # Data loader (input pipeline)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # Show up output tensor.Size
    for i, (images, labels) in enumerate(train_loader):
        print(images.size())


# Using "CustomImageDataset" in SPECT_MPI_step3_dataLoader:
#
#   from SPECT_MPI_step3_dataLoader import CustomImageDataset