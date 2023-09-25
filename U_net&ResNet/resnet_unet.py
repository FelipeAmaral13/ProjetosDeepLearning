from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
from torchvision.models.resnet import resnext50_32x4d
import torch.nn.functional as F

class ResNeXtUNet(nn.Module):
    """
    Uma classe que define uma rede neural para segmentação de imagens, combinando uma ResNet como
    backbone com uma arquitetura U-Net para tarefas de segmentação semântica ou de instâncias.

    Args:
        n_classes (int): O número de classes de segmentação (default: 1).
    """

    def __init__(self, n_classes=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone da ResNet
        self.base_model = resnext50_32x4d(weights="ResNeXt50_32X4D_Weights.DEFAULT")
        self.base_layers = list(self.base_model.children())
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        # Codificadores
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        # Decodificadores
        self.decoder4 = self.decoder_block(filters[3], filters[2])
        self.decoder3 = self.decoder_block(filters[2], filters[1])
        self.decoder2 = self.decoder_block(filters[1], filters[0])
        self.decoder1 = self.decoder_block(filters[0], filters[0])

        # Camadas finais
        self.last_conv0 = self.conv_relu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, x):
        """
        Função forward da rede.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de saída.
        """
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.last_conv0(d1)
        out = self.last_conv1(out)

        out = torch.sigmoid(out)

        return out

    def decoder_block(self, in_channels, out_channels):
        """
        Cria um bloco de decodificador.

        Args:
            in_channels (int): Número de canais de entrada.
            out_channels (int): Número de canais de saída.

        Returns:
            torch.nn.Sequential: Bloco de decodificador.
        """
        return nn.Sequential(
            self.conv_relu(in_channels, in_channels // 4, 1, 0),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4, stride=2, padding=1, output_padding=0),
            self.conv_relu(in_channels // 4, out_channels, 1, 0)
        )

    def conv_relu(self, in_channels, out_channels, kernel_size, padding):
        """
        Cria uma camada de convolução seguida por uma função de ativação ReLU.

        Args:
            in_channels (int): Número de canais de entrada.
            out_channels (int): Número de canais de saída.
            kernel_size (int): Tamanho do kernel de convolução.
            padding (int): Valor de preenchimento para a convolução.

        Returns:
            torch.nn.Sequential: Camada de convolução seguida de ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
    
    def train_model(self, num_epochs, train_loader, learning_rate=0.001, weight_decay=1e-3):
        """
        Treina o modelo.

        Args:
            num_epochs (int): Número de épocas de treinamento.
            train_loader (torch.utils.data.DataLoader): DataLoader para os dados de treinamento.
            learning_rate (float): Taxa de aprendizado para o otimizador (default: 0.001).
            weight_decay (float): Peso para a regularização L2 (default: 1e-3).
        """
        optimizer = torch.optim.Adamax(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        self.to(self.device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            total_iou = 0.0
            total_dice = 0.0

            for batch in tqdm(train_loader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                labels = labels.unsqueeze(1)

                optimizer.zero_grad()

                outputs = self(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calcule a acurácia
                predicted_labels = (outputs > 0.5).float()  # Converta as previsões em rótulos binários
                correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += labels.numel()

                # Calcule o IoU
                intersection = torch.logical_and(predicted_labels, labels).sum().item()
                union = torch.logical_or(predicted_labels, labels).sum().item()
                iou = intersection / union if union > 0 else 0  # Evite a divisão por zero
                total_iou += iou

                # Calcule o Dice
                dice = (2 * intersection) / (predicted_labels.sum().item() + labels.sum().item())
                total_dice += dice

            average_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_samples  # Acurácia calculada como proporção de previsões corretas
            average_iou = total_iou / len(train_loader)  # IoU médio calculado para esta época
            average_dice = total_dice / len(train_loader)  # Dice médio calculado para esta época

            print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f} - IoU: {average_iou:.4f} - Dice: {average_dice:.4f}')


    def predict_image(self, image_path):
        """
        Predição de uma máscara.

        Args:
            image_path (str): Caminho para o arquivo de imagem de entrada.

        Returns:
            torch.Tensor: Máscara prevista como tensor.
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.ANTIALIAS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_mask = self(image)

        return predicted_mask
    

    def save_model(self, filepath):
        """
        Salva o modelo treinado em um arquivo.

        Args:
            filepath (str): Caminho para o arquivo onde o modelo será salvo.
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'n_classes': self.n_classes,
        }
        torch.save(checkpoint, filepath)
        print(f'Model saved to {filepath}')

    @classmethod
    def load_model(cls, filepath, device='cuda'):
        """
        Carrega um modelo previamente salvo a partir de um arquivo.

        Args:
            filepath (str): Caminho para o arquivo onde o modelo foi salvo.
            device (str): Dispositivo onde o modelo será carregado (default: 'cuda').

        Returns:
            ResNeXtUNet: Instância do modelo carregado.
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(n_classes=checkpoint['n_classes'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Model loaded from {filepath}')
        return model
