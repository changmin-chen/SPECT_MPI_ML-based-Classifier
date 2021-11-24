from os.path import join
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from SPECT_MPI_step4_CustomNetwork import CustomNetwork
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch


if __name__ == '__main__':

    # Define transform and/or target_transform (this is optional)
    img_transform = torch.nn.Sequential(
        T.ConvertImageDtype(torch.float),
    )

    # Dataloader
    train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', 'proc_data', 'TrainSet'), transform=img_transform)
    test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', 'proc_data', 'TestSet'), transform=img_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    # Network
    model = CustomNetwork()

   