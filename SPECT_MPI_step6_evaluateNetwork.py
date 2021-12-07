from os.path import join
import torch
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader

# Select image processing version dataset & corresponding network

# Version 0: Not sure, most of the predicted result were abnormal (i.e. too sensitive)
# ver = 'proc_data_ver0'
# model_name = 'Net_epoch30_bthsize16_lr5e-3_ver0.pth'

# Version 1: this model is likey more intelligently to classify normals & abnormals
# ver = 'proc_data_ver1'
# model_name = 'Net_epoch30_bthsize16_lr5e-3_ver1.pth'

# Version 2: model all guess Abnormal
ver = 'proc_data_ver2'
model_name = 'Net_epoch30_bthsize16_lr5e-3_ver2.pth'

# Evaluate the network using test dataset
model = torch.load(model_name).cuda()
test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'))
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
            images, labels = images.cuda(), labels.cuda()
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
        
