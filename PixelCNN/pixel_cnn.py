import torch
import torch.nn as nn

class MaskedConvolution(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConvolution, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConvolution, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1), padding='same', bias=False),
            nn.ReLU(inplace=True),
            MaskedConvolution('B', out_channels // 2, out_channels // 2, kernel_size=(3, 3), padding='same', bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(1, 1), padding='same', bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return x + self.layers(x)

class PixelCNN(nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.input_conv = nn.Sequential(
            MaskedConvolution('A', 1, 128, kernel_size=(7, 7), stride=(1, 1), padding='same', bias=False),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.Sequential(
            *[ResidualBlock(128, 128) for _ in range(5)]
        )
        self.last_conv = nn.Sequential(
            MaskedConvolution('B', 128, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            MaskedConvolution('B', 128, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return self.out(x)


