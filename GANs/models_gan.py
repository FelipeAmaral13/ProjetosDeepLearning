import torch.nn as nn

class ModelGan(nn.Module):

    def __init__(self, ngpu, num_channels=3, noise_dimension=100, generator_feature_maps=64, discriminator_feature_maps=64) -> None:
        super(ModelGan, self).__init__()
        self.num_channels = num_channels
        self.noise_dimension = noise_dimension
        self.generator_feature_maps = generator_feature_maps
        self.discriminator_feature_maps = discriminator_feature_maps
        self.ngpu = ngpu
    
    def generator_arch(self):

        arch = nn.Sequential(            
            nn.ConvTranspose2d(self.noise_dimension, self.generator_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.generator_feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_maps * 8, self.generator_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_maps * 4, self.generator_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_maps * 2, self.generator_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.generator_feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.generator_feature_maps, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        return arch

    def discriminator_arch(self):
        
        arch = nn.Sequential(
            nn.Conv2d(self.num_channels, self.discriminator_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_maps, self.discriminator_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_maps * 2, self.discriminator_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_maps * 4, self.discriminator_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.discriminator_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.discriminator_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        return arch
