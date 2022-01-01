from os.path import join
import pandas as pd
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as T
import utils.MPIdataset as mpidataset
import utils.Fake3DNet as net


# Option setting
# (1) image processing version
ver = 'proc_data_ver1'

# (2) model
model = net.Fake3DNet_Conv2d()

# (3) hyperparameters
args = {'num_epochs': 5,
        'batch_size': 16,
        'learning_rate': 0.001}

# (4) loss function
loss_func = nn.CrossEntropyLoss()

# (5) optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])


def labels_preproc(source_dir='GroundTruth.xls', target_dir='.'):
    """
    Objective:
    Preprocessing the sample labels
    (1) replace the original filenames into the processed filenames (.jpg -> .nii)
    (2) replace the string-labels into the number-labels: (Normal-> 0, Anormal-> 1)
    (3) resorting the dataset: TrainSet unchanged, but adjoin the TestSet and ValidationSet

    Output:
    write trainSet.csv and testSet.csv files
    """

    data = pd.read_excel(source_dir)
    data = data.iloc[0:192]
    data = data[['# Patients','Evaluation','Age','Sex','Set Distribution']]
    data['# Patients'] = data['# Patients'].apply(str) + '.nii'
    data['Evaluation'] = data['Evaluation'].transform(lambda x: 0 if x == 'Normal' else 1) 
    data['Set Distribution'] = data['Set Distribution'].replace('Validation', 'Train')
    trainSet = data.loc[data['Set Distribution'] == 'Train']
    testSet = data.loc[data['Set Distribution'] == 'Test']

    trainSet.to_csv(join(target_dir, 'trainSet.csv'), header=True, index=False)
    testSet.to_csv(join(target_dir, 'testSet.csv'), header=True, index=False)


def dataset_statistics(ver):
    """
    Objective:
    Compute mean & std for each channel, for Standardization of the dataset.

    if your RAM can load all samples at once (like dataset for this project), the code should be much shorter
    however, the following code assumed that it maynot be the case
    """
    dataset = mpidataset.MPIdataset('trainSet.csv', join('..', ver, 'TrainSet'))

    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
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

# Get mean and standard deviation for standardizing the images
mean, std = dataset_statistics(ver)
def standardization(image, mean=mean, std=std):
    return (image - mean[:, None, None, None])*(1./std[:, None, None, None])


def train(ver, model, loss_func, optimizer, args):
 
    # Preprocess the sample labels
    labels_preproc()

    # TrainSet dataloader, utilize WeightedRandomSampler for unbalance data
    train_dataset = mpidataset.MPIdataset('trainSet.csv', join('..', ver, 'TrainSet'), transform=standardization)
    weights = 1. /torch.tensor([sum(train_dataset.labels.iloc[:, 1]==0), sum(train_dataset.labels.iloc[:,1]==1)], dtype=torch.float32)
    train_target = torch.tensor(train_dataset.labels.iloc[:, 1], dtype=torch.long)
    sample_weights = weights[train_target]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], pin_memory=True, sampler=sampler)

    # TestSet dataloader, for in-time model performance validation
    test_dataset = mpidataset.MPIdataset('testSet.csv', join('..', ver, 'TestSet'), transform=standardization)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], pin_memory=True, shuffle=False)

    # Select device, use gpu if available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = model.to(device)
    loss = loss_func.to(device)

    # Define model performance measurement
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
            for _, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                test_loss.append(loss.item())

        # Evaluate model performance
        acccuracy_test = classification_accuracy(test_loader, model)
        print('Epoch {}, Time {:.2f}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
              .format(epoch + 1, time.time() - tini,
                      sum(train_loss) / len(train_loss), sum(test_loss) / len(test_loss), acccuracy_test))
    
    
    # Save the trained model
    model = model.cpu()
    model_name = 'Net_epoch{}_Batch{}_lr{}'.format(args['num_epochs'], args['batch_size'], args['learning_rate'])
    torch.save(model, model_name+'.pth')


if __name__=='__main__':
    train(ver, model, loss_func, optimizer, args)