import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def append_parameters(blocks):
    parameters = [list(x.parameters()) for x in blocks]
    all_parameters = []
    for pars in parameters:
        for par in pars:
            all_parameters.append(par)
    return all_parameters


def to_freeze(pars):
    for par in pars:
        par.requires_grad = False


def to_unfreeze(pars):
    for par in pars:
        par.requires_grad = True


class ResnetFeatures(nn.Module):
    def __init__(self, resnet_name, pretrained):
        super(ResnetFeatures, self).__init__()

        self.resnet = getattr(models, resnet_name)(pretrained=pretrained)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        to_freeze(list(self.resnet.parameters()))
        pars = append_parameters([getattr(self.resnet, x) for x in ['layer4']])
        to_unfreeze(pars)

        print_num_of_parameters(self.resnet)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.shape[0], 512, 12, 12)
        return x


class modelPretrained(nn.Module):
    def __init__(self, args_m):
        super(modelPretrained, self).__init__()
        if args_m.backbone.startswith('res'):
            self.features = ResnetFeatures(args_m.backbone, pretrained=args_m.pretrained)
        elif args_m.backbone == 'SqueezeNet':
            self.features = getattr(models, args_m.backbone)().features
        else:
            self.features = getattr(models, args_m.backbone)(pretrained=args_m.pretrained).features

        if args_m.backbone == 'alexnet':
            self.fmap_c = 256
        elif args_m.backbone == 'densenet121':
            self.fmap_c = 1024
        else:
            self.fmap_c = 512

        # fusion part
        self.classifier = nn.Conv2d(self.fmap_c, args_m.n_classes, 1, 1, 0)
        self.classifier_cat = nn.Conv2d(self.fmap_c * 23, args_m.n_classes, 1, 1, 0)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fuse = args_m.fuse

    def forward(self, x):   # (B, 3, 224, 224, 23)
        x = x[0]  # for single slice
        # dummies
        out = None  # output of the model
        features = None  # features we want to further analysis
        # reshape
        B = x.shape[0]
        x = x.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x = x.reshape(B * x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # (B*23, 3, 224, 224)
        # features
        x = self.features(x)  # (B*23, 512, 7, 7)

        # fusion
        if self.fuse == 'cat':  # concatenate across the slices
            x = self.avg(x)  # (B*23, 512, 1, 1)
            x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 23, 512, 1, 1)
            xcat = x.view(B, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # (B, 23*512, 1, 1)
            out = self.classifier_cat(xcat)  # (Classes)
            out = out[:, :, 0, 0]
            features = (xcat)

        if self.fuse == 'max':  # max-pooling across the slices
            x = self.avg(x)  # (B*23, 512, 1, 1)
            x = x.view(B, x.shape[0] // B, x.shape[1], x.shape[2], x.shape[3])  # (B, 23, 512, 1, 1)
            features, _ = torch.max(x, 1)  # (B, 512, 1, 1)
            out = self.classifier(features)  # (Classes)
            out = out[:, :, 0, 0]
        return out, features