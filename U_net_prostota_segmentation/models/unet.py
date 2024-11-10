import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .conv_block import Conv2dBlock

class UNetWithResNetBackbone(nn.Module):
    def __init__(self, n_classes=1, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNetWithResNetBackbone, self).__init__()
        self.input_conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = models.resnet34(pretrained=True)
        self.enc1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.enc2, self.enc3 = self.resnet.layer1, self.resnet.layer2
        self.enc4, self.enc5 = self.resnet.layer3, self.resnet.layer4
        self.u6, self.c6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), Conv2dBlock(512, 256, batchnorm)
        self.u7, self.c7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), Conv2dBlock(256, 128, batchnorm)
        self.u8, self.c8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), Conv2dBlock(128, 64, batchnorm)
        self.u9, self.c9 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), Conv2dBlock(128, 32, batchnorm)
        self.out = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x)
        c1 = self.enc1(x)  
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)
        
        u6 = F.interpolate(self.u6(c5), size=c4.shape[2:])  
        c6 = self.c6(torch.cat([u6, c4], dim=1))
        
        u7 = F.interpolate(self.u7(c6), size=c3.shape[2:])  
        c7 = self.c7(torch.cat([u7, c3], dim=1))
        
        u8 = F.interpolate(self.u8(c7), size=c2.shape[2:])  
        c8 = self.c8(torch.cat([u8, c2], dim=1))
        
        u9 = F.interpolate(self.u9(c8), size=c1.shape[2:])  
        c9 = self.c9(torch.cat([u9, c1], dim=1))
        
        out = self.out(c9)
        out = F.interpolate(out, size=x.shape[2:])  
        
        return torch.sigmoid(out)

    # Método para salvar o modelo
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Modelo salvo em: {file_path}")

    # Método para carregar o modelo salvo
    def load_model(self, file_path, device="cpu"):
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        print(f"Modelo carregado de: {file_path}")
