
from torch.nn.modules.linear import Linear
import torchvision
import torch
import torch.nn as nn


class Fake3DNet(nn.Module):
    def __init__(self):
        super(Fake3DNet, self).__init__()

        model_pretrained = torchvision.models.vgg16(pretrained=True)

        # Section 1: Define model feature
        # Create a fake 3D-Net (because the operations are actually is 2D),
        # which using the pretrained parameters in VGG16
        self.features = model_pretrained.features
        for i, layer in enumerate(self.features.children()):
            if isinstance(layer, nn.Conv2d):
                in_ch, out_ch = layer.in_channels, layer.out_channels
                layer = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
                self.features[i] = layer  

            elif isinstance(layer, nn.MaxPool2d):
                layer = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1), padding=0, dilation=1, ceil_mode=False)
                self.features[i] = layer

        # We don't have that much data
        for par in self.features.parameters():
            par.requires_grad = False

        # Secition 2: Define model classifier
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2*40, 2048),
            nn.Linear(2048, 2),
            nn.Softmax(dim=-1)
        )

    # Section 3: Define forward function
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1) # concatenate all features from all planes
        x = self.classifier(x)
        return x



# Test
if __name__ == "__main__":
    model = Fake3DNet()
    x = torch.rand([1, 3, 89, 89, 40])
    y = model.forward(x)
    print(y)