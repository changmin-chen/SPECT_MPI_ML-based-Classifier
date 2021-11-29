import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.nn import functional as F

class CrossEntropyLoss(nn.Module):
    """Dice loss of binary class
    Args:
    Returns:
        Loss tensor
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss_c = nn.CrossEntropyLoss()(x, y)
        _, classification_pred = torch.max(x, 1)
        #acc = (classification_pred == target).sum().type(torch.FloatTensor)

        return loss_c,


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss()

    def __len__(self):
        """ length of the components of loss to display """
        return 1

    def forward(self, output, labels):
        x = output[0]
        y = labels[0]
        loss_classify, = self.cross_entropy_loss(x, y)

        loss_all = [loss_classify]
        loss_val = loss_all[0]

        return loss_val, loss_all


class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, true_masks, out):
        n_classes = out.shape[1]
        masks_probs = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        masks_probs = masks_probs.reshape(masks_probs.shape[0] * masks_probs.shape[1] * masks_probs.shape[2],
                                          masks_probs.shape[3])  # (B * H * W, C)
        _, masks_pred = torch.max(masks_probs, 1)

        dice = np.zeros(n_classes)
        dice_tp = np.zeros(n_classes)
        dice_div = np.zeros(n_classes)
        for c in range(n_classes):
            dice_tp[c] += ((masks_pred == c) & (true_masks.view(-1) == c)).sum().item()
            dice_div[c] += ((masks_pred == c).sum().item() + (true_masks.view(-1) == c).sum().item())
            dice[c] = 2 * dice_tp[c] / dice_div[c]

        return dice[:]  # omit the background channel