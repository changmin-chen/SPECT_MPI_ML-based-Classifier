import torchvision
import torch.nn as nn

class CustomNetwork(nn.Module): 
    def __init__(self, pretrained=True):
        super(CustomNetwork, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.features[0] = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) # replace
        self.classifier = torchvision.models.vgg16(pretrained=pretrained).classifier
        self.classifier[-2] = nn.Linear(4096, 512, bias=True) # replace
        self.classifier[-1] = nn.Linear(512, 2, bias=True) # replace

        for par in list(self.features[1:].parameters()):
            par.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x