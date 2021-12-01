from os.path import join

from torch.optim import optimizer
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from SPECT_MPI_step4_fake3DNet import Fake3DNet
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn

# Step 1: Select image processing version, and commend the others
ver = 'proc_data_ver0'
# ver = 'proc_data_ver1'
# ver = 'proc_data_ver2'

# Step 2: Hyperparameters, Loss function, Optimizer
args = {'num_epochs': 2,
        'batch_size': 16,
        'learning_rate': 0.001}

# Step 3: Define training function
def train(model, args, train_loader, test_loader, loss_function, optimizer):
    for epoch in range(args['num_epochs']):
        train_loss = []

        for _, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 
            train_loss.append(loss.item())

        test_loss = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                outputs = model(images)
                loss = loss_function(outputs, labels)
                test_loss.append(loss.item())

# Step 4: Training
if __name__ == '__main__':

    # Dataloader
    transforms = torch.nn.Sequential(
        T.ConvertImageDtype(torch.float32)
    )
    train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', ver, 'TrainSet'), transform=transforms)
    test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'), transform=transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False)

    # Get the predifined model
    model = Fake3DNet()

    # Loss function, Optimizer
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

    # Start training
    train(model, args, train_loader, test_loader, loss, opt)
    torch.save(model, 'Happy'+'.pth')