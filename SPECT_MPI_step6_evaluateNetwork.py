from os.path import join
import torch
from SPECT_MPI_step3_CustomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader

# Select image processing version dataset & corresponding network

# Version 0: Not sure, most of the predicted result guess Abnormal
# ver = 'proc_data_ver0'
# model_name = 'Net_ver0_epoch20_trL03_teL04_cros_sgd.pth'

# Version 1: this model is likey more intelligently to classify b/w normals & abnormals
ver = 'proc_data_ver1'
model_name = 'Net_ver1_epoch20_trL04_teL05_Ac08_cros_sgd.pth'

# Version 2: model all guess Abnormal
# ver = 'proc_data_ver2'
# model_name = 'Net_ver2_epoch20_trL03_teL04_Ac078_cros_sgd.pth'


# Evaluate the network using test dataset
model = torch.load(model_name).cuda()
test_dataset = CustomImageDataset(join('..', 'testSet.csv'), join('..', ver, 'TestSet'))
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)

if __name__ == '__main__': 
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(f'The predicted classes in test dataset: {predicted}')
        print(f'The groundtruth classes in test dataset: {labels}')
        
