# models/conv_block.py
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)) if self.batchnorm else self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)) if self.batchnorm else self.conv2(x))
        return x
