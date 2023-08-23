import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelo.pixel_cnn_2 import PixelCNN
from tqdm import tqdm

class TreinPixelCNN:
    def __init__(self):
        self.model = PixelCNN()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.train_dataset = datasets.FashionMNIST(root = "dados/", train = True, download = True, transform = self.transform )
        self.eval_dataset = datasets.FashionMNIST(root="dados/", train=False, download=True, transform=self.transform )


    def train_model(self, batch_size=32, learning_rate=0.0005, num_epochs=41, patience=5):
        
        
        train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        bce_loss = nn.BCELoss()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            mean_loss = 0
            for image, _ in tqdm(train_loader):
                image = image.to(self.device)
                image = torch.where(image == 0., 0., 1.)
                optimizer.zero_grad()
                outputs = self.model(image)
                loss = bce_loss(outputs, image)
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
            mean_loss /= len(train_loader)
            print(f"Epoch: {epoch:>3d} Erro do Modelo: {mean_loss / len(train_loader):>6f}")

            if mean_loss < best_loss:
                best_loss = mean_loss
                patience_counter = 0
                self.save_model(epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping na epoca {epoch}.")
                    break

            if epoch % 5 == 0:
                self.save_model(epoch)
    
    def evaluate_model(self, path_model, batch_size=32):
        self.model.load_state_dict(torch.load(path_model, map_location = self.device))
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        eval_loader = DataLoader(self.eval_dataset, batch_size = batch_size, shuffle = True)
        bce_loss = nn.BCELoss()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for image, _ in tqdm(eval_loader):
                image = image.to(self.device)
                image = torch.where(image == 0., 0., 1.)
                outputs = self.model(image)
                loss = bce_loss(outputs, image)
                total_loss += loss.item() * image.size(0)
                num_samples += image.size(0)
        
        average_loss = total_loss / num_samples
        print(f"Average Loss on Evaluation Set: {average_loss:.6f}")
                

    def save_model(self, epoch):
        save_path = os.path.join(os.getcwd(), "modelo_pixelcnn{}.pt".format(epoch))
        torch.save(self.model.state_dict(), save_path)
    
    def predict_image(self, path_model):
        self.model.load_state_dict(torch.load(path_model, map_location = self.device))
        self.model.to(self.device)

        imagem_gerada = np.zeros((16, 1, 64, 64), dtype = np.float32)
        imagem_gerada = torch.from_numpy(imagem_gerada)
        imagem_gerada = imagem_gerada.to(self.device)

        with torch.no_grad():
            for h in tqdm(range(28)):
                for w in range(28):
                    previsao = self.model(imagem_gerada)
                    pixel_previsto = torch.bernoulli(previsao[:, :, h, w])
                    imagem_gerada[:, :, h, w] = pixel_previsto
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(imagem_gerada[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.show()


train = TreinPixelCNN()
train.train_model()