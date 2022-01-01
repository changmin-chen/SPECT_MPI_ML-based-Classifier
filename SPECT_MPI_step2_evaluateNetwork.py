from os.path import join
import torch
import utils.MPIdataset as mpidataset
from torch.utils.data import DataLoader
from SPECT_MPI_step1_trainNetwork import dataset_statistics


# Select image processing version dataset & corresponding trained network
ver = 'proc_data_ver1'
model_name = 'Net_epoch5_Batch16_lr0.001.pth'


def main(ver, model_name):

    # Selecting device, use gpu if available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Define standardization, mean & std calculation is based on TrainSet
    test_dataset = mpidataset.MPIdataset('testSet.csv', join('..', ver, 'TestSet'))
    mean, std = dataset_statistics(ver)
    def standardization(image, mean=mean, std=std):
        return (image - mean[:, None, None, None])*(1./std[:, None, None, None])

    # Evaluate the network using test dataset
    model = torch.load(model_name).to(device)
    test_dataset = mpidataset.MPIdataset('testSet.csv', join('..', ver, 'TestSet'), transform=standardization)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


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


if __name__ == '__main__':
    main(ver=ver, model_name=model_name)