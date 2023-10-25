import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterDistLoss(nn.Module):
    def __init__(self, config=None):
        super(CenterDistLoss, self).__init__()

    def forward(self, z, center):
        dist = torch.sum((z - center) ** 2, dim=1)
        # return torch.sqrt(dist)
        return dist, 1/dist

