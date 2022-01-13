import torchvision
import torch
import torch.nn as nn


# SwitchNorm
# code copied from https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py
class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


# Extract shallower features.
class Fake3DNet_Conv2d(nn.Module):
    def __init__(self):
        super(Fake3DNet_Conv2d, self).__init__()
    
        model_pretrained = torchvision.models.vgg16(pretrained=True)

        # Section 1: Define model feature
        self.features = model_pretrained.features[:10]

        # We don't have that much data
        for par in self.features.parameters():
            par.requires_grad = False

        # Secition 2: Define model classifier
        self.classifier = nn.Sequential(
        nn.Linear(128*22*22*40, 64),
        nn.ReLU(),
        SwitchNorm1d(64),
        nn.Linear(64, 2)
        )

    # Section 3: Define forward function
    def forward(self, x):
        (B, C, H, W, D) = x.shape

        # temporarily change the input format from [B, C, H, W, D] into [B*D, C, H, W], then perfrom feature extration
        # use torch.reshape instead of torch.view, because of the memory non-contiguousity
        # after permute: torch.Tensor.is_contiguous(x) >> False, after reshape: torch.Tensor.is_contiguous(x) >> True
        x = x.permute([0, 4, 1, 2, 3])
        x = x.reshape(B*D, C, H, W) 
        x = self.features(x)

        # after feature extration, layout the (D,C,H,W)
        x = x.view(B, -1)

        # perfrom classification
        x = self.classifier(x)
        return x


# Extract deeper features. Not recommended, may be overfitting
class Fake3DNet_Conv2d_Deep(nn.Module):
    def __init__(self):
        super(Fake3DNet_Conv2d_Deep, self).__init__()
    
        model_pretrained = torchvision.models.vgg16(pretrained=True)

        # Section 1: Define model feature
        self.features = model_pretrained.features

        # We don't have that much data
        for par in self.features.parameters():
            par.requires_grad = False

        # Secition 2: Define model classifier
        self.classifier = nn.Sequential(
        nn.Linear(512*2*2*40, 64),
        nn.LeakyReLU(),
        SwitchNorm1d(64),
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        SwitchNorm1d(64),
        nn.Linear(64, 2),
        )

    # Section 3: Define forward function
    def forward(self, x):
        (B, C, H, W, D) = x.shape

        # temporarily change the input format from [B, C, H, W, D] into [B*D, C, H, W], then perfrom feature extration
        # use torch.reshape instead of torch.view, because of the memory non-contiguousity
        # after permute: torch.Tensor.is_contiguous(x) >> False, after reshape: torch.Tensor.is_contiguous(x) >> True
        x = x.permute([0, 4, 1, 2, 3])
        x = x.reshape(B*D, C, H, W) 
        x = self.features(x)

        # after feature extration, layout the (D,C,H,W)
        x = x.view(B, -1)

        # perfrom classification
        x = self.classifier(x)
        return x


# Test
if __name__ == "__main__":
    
    x = torch.randn([2, 3, 89, 89, 40])

    # For "Fake3DNet_Conv2d"
    model = Fake3DNet_Conv2d()
    print('testing Fake3DNet_Conv2d:\n')
    with torch.no_grad():
        for _ in range(1):
            y = model(x)
            print(y)     