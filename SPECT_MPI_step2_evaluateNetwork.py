from os.path import join
import torch
import utils.MPIdataset as mpidataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from SPECT_MPI_step1_trainNetwork import dataset_statistics


# Select image processing version dataset & corresponding trained network
ver = 'proc_data_ver0'
model_name = 'Net_epoch20_Batch16_lr0.001.pth'

def draw_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )

    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


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
        all_scores = []
        all_predicted = []
        all_labels = []

        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            scores, predicted = torch.max(outputs.data, 1)

            # Append output
            all_scores.append(scores)
            all_predicted.append(predicted)
            all_labels.append(labels)
            
        all_scores = torch.cat(all_scores).cpu()
        all_predicted = torch.cat(all_predicted).cpu()
        all_labels = torch.cat(all_labels).cpu()

        # Accuracy
        total = all_labels.size(0)
        correct = (all_predicted == all_labels).sum()
        
        # Sensitivity, Specificity
        TP = torch.logical_and(all_labels.type(torch.bool)==True, all_predicted.type(torch.bool)==True).sum()
        TN = torch.logical_and(all_labels.type(torch.bool)==False, all_predicted.type(torch.bool)==False).sum()
        FP = torch.logical_and(all_labels.type(torch.bool)==False, all_predicted.type(torch.bool)==True).sum()
        FN = torch.logical_and(all_labels.type(torch.bool)==True, all_predicted.type(torch.bool)==False).sum()
                
        # Display evaluations
        print(f'The predicted classes in test dataset: {all_predicted.cpu()}')
        print(f'The groundtruth classes in test dataset: {all_labels.cpu()}')
        accuracy = correct.cpu().numpy() / total
        sensitivity = TP / (TP+FN)
        specificity = TN / (FP+TN)
        print(f'Total Samples Number:{total}, Accuracy:{accuracy:.4f}, Sensitivity:{sensitivity:.4f}, Specificity:{specificity:.4f}')
        draw_roc_curve(all_labels, all_predicted)


if __name__ == '__main__':
    main(ver=ver, model_name=model_name)