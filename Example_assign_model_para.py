import torchvision
import torch
import torch.nn as nn

# Read VGG16
model = torchvision.models.vgg16(pretrained=True).features
layer = model[0]

# Assign Model Parameters
weights = layer.weight
bias = layer.bias
print(f'Convolution layer parameters shape: {weights.shape}')

# appending the dimension
weights = torch.unsqueeze(weights, dim=-1)
print(f'Convolution layer parameters shape appended: {weights.shape}')
new_lr = nn.Conv3d(3, 64, kernel_size=(3,3,1), stride=(1,1,1))
new_lr.weight = nn.parameter.Parameter(weights)
new_lr.bias = nn.parameter.Parameter(bias)

for par in new_lr.parameters():
    print(par.shape)

input = torch.rand([1, 3, 89, 89, 40]) #torch.Size([B, C, H, W, P])
output = new_lr(input)
print(f'Output shape={output.shape}')