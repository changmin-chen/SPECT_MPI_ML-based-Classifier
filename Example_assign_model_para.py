import torchvision
import torch
import torch.nn as nn

# Read VGG16
model = torchvision.models.vgg16(pretrained=True).features
lr = model[0]

# Assign Model Parameters
wb = list(lr.parameters())
weights = wb[0]
bias = wb[1]
print(f'Convolution layer parameters shape: {weights.shape}')
weights = torch.unsqueeze(weights, dim=-1)
print(f'Convolution layer parameters shape appended: {weights.shape}')
new_lr = nn.Conv3d(3, 64, kernel_size=(3,3,1))
new_lr.weight = nn.parameter.Parameter(weights)