from os.path import join
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from SPECT_MPI_step4_fake3DNet import Fake3DNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time

# Training device, use gpu if available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Step 1: Select image processing version, and commend the others
# ver = 'proc_data_ver0'
ver = 'proc_data_ver1'
# ver = 'proc_data_ver2'

# Step 2: Hyperparameters, Loss function, Optimizer
args = {'num_epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.005}

# Step 3: Define measurement function
def classification_accuracy(test_loader, model):
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = correct.cpu().numpy() / total
        return accuracy

# Step 3: Training
if __name__ == '__main__':

    # Train dataloader, utilize WeightedRandomSampler for unbalance data
    train_dataset = CustomImageDataset(join('..', 'trainSet.csv'), join('..', ver, 'TrainSet'))
    weights = 1. /torch.tensor([sum(train_dataset.labels.iloc[:, 1]==0), sum(train_dataset.labels.iloc[:,1]==1)], dtype=torch.float32)
    train_target = torch.tensor(train_dataset.labels.iloc[:, 1], dtype=torch.long)
    sample_weights = weights[train_target]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)

    # Test dataloader
    test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    # Get the predifined model
    model = Fake3DNet().to(device)

    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

    # Start training
    for epoch in range(args['num_epochs']):

        tini = time.time()
        train_loss = []
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() 
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()  
            optimizer.step()
            train_loss.append(loss.item())
        
        test_loss = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                test_loss.append(loss.item())

        # calculate the accuracy
        acccuracy_test = classification_accuracy(test_loader, model)
        print('Epoch {}, Time {:.2f}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
              .format(epoch + 1, time.time() - tini,
                      sum(train_loss) / len(train_loss), sum(test_loss) / len(test_loss), acccuracy_test))
    
    
    # Save the trained model
    model = model.cpu()
    model_name = 'Net_epoch{}_Batch{}_lr{}'.format(args['num_epochs'], args['batch_size'], args['learning_rate'])
    torch.save(model, model_name+'.pth')