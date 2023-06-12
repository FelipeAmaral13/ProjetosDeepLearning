import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import logging

from models_gan import ModelGan


class FaceGan:
    def __init__(self):
        """
        Initializes the FaceGan class.
        """
        manualSeed = 999
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        self.ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.dataroot = os.path.join("dados", "celeba")
        self.image_size = 64
        self.dataset = None
        self.dataloader = None
        self.real_batch = None
        self.load_dataset()
        self.model_gan = ModelGan(self.ngpu)
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """
        Set up the logger configuration.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(levelname)s: %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def load_dataset(self):
        """
        Loads the dataset and creates a dataloader.
        """
        dataset = dset.ImageFolder(
            root=self.dataroot,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )
        self.real_batch = next(iter(self.dataloader))
        self.plot_samples()

    def plot_samples(self):
        """
        Plots a grid of real training images.
        """
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    self.real_batch[0].to(self.device)[:64],
                    padding=2,
                    normalize=True
                ).cpu(),
                (1, 2, 0)
            )
        )
        plt.show()

    def init_pesos(self, m):
        """
        Initializes the weights of the neural network module.

        Args:
            m (nn.Module): Neural network module.
        """
        classname = m.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def load_models(self):
        """
        Loads the generator and discriminator models and initializes their weights.
        """
        self.dsanetG = self.model_gan.generator_arch().to(self.device)
        self.dsanetG.apply(self.init_pesos)
        print(self.dsanetG)
        self.dsanetD = self.model_gan.discriminator_arch().to(self.device)
        self.dsanetD.apply(self.init_pesos)
        print(self.dsanetD)

    def train_model(self):
        """
        Trains the GAN model.

        Returns:
            G_losses (list): List of generator losses during training.
            D_losses (list): List of discriminator losses during training.
            img_list (list): List of generated images at certain intervals.
        """
        criterion = nn.BCELoss()
        lr = 0.0002
        beta1 = 0.5
        beta2 = 0.999
        optimizerD = optim.Adam(self.dsanetD.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerG = optim.Adam(self.dsanetG.parameters(), lr=lr, betas=(beta1, beta2))
        num_epochs = 1
        real_label = 1.0
        fake_label = 0.0
        fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        self.logger.info("Training started...")

        for epoch in range(num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                self.dsanetD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.dsanetD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                noise = torch.randn(b_size, 100, 1, 1, device=self.device)
                fake = self.dsanetG(noise)
                label.fill_(fake_label)
                output = self.dsanetD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                self.dsanetG.zero_grad()
                label.fill_(real_label)
                output = self.dsanetD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                if i % 50 == 0:
                    self.logger.info(
                        "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f" 
                        % (epoch, num_epochs, i, len(self.dataloader), errD.item(), errG.item())
                        )
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.dsanetG(fixed_noise).detach().cpu()

                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

        self.logger.info("Training started...")
        return G_losses, D_losses, img_list

    def save_models(self, save_path):
        """
        Saves the generator and discriminator models to the specified path.

        Args:
            save_path (str): Path to save the models.
        """
        os.makedirs(save_path, exist_ok=True)

        generator_path = os.path.join(save_path, "generator.pth")
        discriminator_path = os.path.join(save_path, "discriminator.pth")

        torch.save(self.dsanetG.state_dict(), generator_path)
        torch.save(self.dsanetD.state_dict(), discriminator_path)

        self.logger.info("Models saved at:", save_path)


    def load_generator_model(self, model_path, num_samples=64):
        """
        Loads the generator model from the specified path and generates samples.

        Args:
            model_path (str): Path to the saved generator model.
            num_samples (int): Number of samples to generate. Defaults to 64.
        """
        self.logger.info("Load Generator Model")
        generator_path = os.path.join(model_path, "generator.pth")
        self.dsanetG = self.model_gan.generator_arch().to(self.device)
        self.dsanetG.load_state_dict(torch.load(generator_path))
        self.dsanetG.eval()
        noise = torch.randn(num_samples, 100, 1, 1, device=self.device)
        samples = self.dsanetG(noise).detach().cpu()
        grid = vutils.make_grid(samples, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Samples")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()

    def evaluate(self, G_losses, D_losses, img_list):
        """
        Displays the evaluation plots and images.

        Args:
            G_losses (list): List of generator losses during training.
            D_losses (list): List of discriminator losses during training.
            img_list (list): List of generated images during training.
        """
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    self.real_batch[0].to(self.device)[:64],
                    padding=5,
                    normalize=True
                ).cpu(),
                (1, 2, 0)
            )
        )
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()


if __name__ == "__main__":
    path_models = os.path.join("models")
    gan = FaceGan()
    G_losses, D_losses, img_list = gan.train_model()
    gan.evaluate(G_losses, D_losses, img_list)
    gan.save_models(path_models)
    gan.generate_samples()
    gan.load_generator_model(path_models)