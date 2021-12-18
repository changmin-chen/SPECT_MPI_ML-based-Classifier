from os.path import join
import torch
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from SPECT_MPI_step3_CustomImageDataset import dataset_statistics
from torch.utils.data import DataLoader

# Selecting device, use gpu if available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Select image processing version dataset & corresponding network
ver = 'proc_data_ver1'
model_name = 'Net_epoch5_Batch16_lr0.001.pth'

# Define transform
# if you had standardize data during training set, specify tranform = standardization,
# if not, specify transform = None
test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'))
mean, std = dataset_statistics(test_dataset)
def standardization(image, mean=mean, std=std):
    # image fromat = [C, H, W, D]
    # image[channel] = (image[channel] - mean[channel]) ./ std[channel]
    image = (image - mean[:, None, None, None])*(1./std[:, None, None, None])

    return image

transform = standardization
# transform = None


# Evaluate the network using test dataset
model = torch.load(model_name).to(device)
test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'), transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)



if __name__ == '__main__': 
    with torch.no_grad():
        total = 0
        correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # Sensitivity, Specificity
            TP += torch.logical_and(labels.type(torch.bool)==True, predicted.type(torch.bool)==True).sum()
            TN += torch.logical_and(labels.type(torch.bool)==False, predicted.type(torch.bool)==False).sum()
            FP += torch.logical_and(labels.type(torch.bool)==False, predicted.type(torch.bool)==True).sum()
            FN += torch.logical_and(labels.type(torch.bool)==True, predicted.type(torch.bool)==False).sum()
                
            # Display prediction
            print(f'The predicted classes in test dataset: {predicted.cpu()}')
            print(f'The groundtruth classes in test dataset: {labels.cpu()}')
        accuracy = correct.cpu().numpy() / total
        sensitivity = TP / (TP+FN)
        specificity = TN / (FP+TN)
        print(f'Total Samples Number:{total}, Accuracy:{accuracy:.4f}, Sensitivity:{sensitivity:.4f}, Specificity:{specificity:.4f}')
        
