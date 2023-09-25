import torch.nn as nn
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

class UNet(nn.Module):
    """
    Implementação da arquitetura UNet para segmentação de imagens.

    Args:
        n_classes (int): Número de classes a serem segmentadas.
    """

    def __init__(self, n_classes, in_channels=3, num_filters=64):
        """
        Inicializa a rede UNet.

        Args:
            n_classes (int): Número de classes a serem segmentadas.
            in_channels (int): Número de canais da imagem de entrada (default: 3 para imagens coloridas RGB).
            num_filters (int): Número de filtros iniciais (default: 64).
        """
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Camadas de convolução descendentes (Encoder)
        self.conv_down1 = self.double_conv(in_channels, num_filters)
        self.conv_down2 = self.double_conv(num_filters, num_filters * 2)
        self.conv_down3 = self.double_conv(num_filters * 2, num_filters * 4)
        self.conv_down4 = self.double_conv(num_filters * 4, num_filters * 8)

        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Camadas de convolução ascendentes (Decoder)
        self.conv_up3 = self.double_conv(num_filters * (4 + 8), num_filters * 4)
        self.conv_up2 = self.double_conv(num_filters * (2 + 4), num_filters * 2)
        self.conv_up1 = self.double_conv(num_filters * (2 + 1), num_filters)

        # Última camada de convolução para ajustar o número de classes
        self.last_conv = nn.Conv2d(num_filters, self.n_classes, kernel_size=1)

    def forward(self, x):
        """
        Realiza uma passagem para a frente na rede UNet.

        Args:
            x (torch.Tensor): Tensor de entrada com as imagens.

        Returns:
            torch.Tensor: Tensor de saída com as máscaras segmentadas.
        """
        # Convoluções descendentes e Max pooling
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)

        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)

        out = self.last_conv(x)
        out = torch.sigmoid(out)

        return out
    
    def double_conv(self, in_channels, out_channels):
        """
        Define uma sequência de duas camadas de convolução seguidas por ativações ReLU.

        Args:
            in_channels (int): Número de canais de entrada.
            out_channels (int): Número de canais de saída.

        Returns:
            nn.Sequential: Sequência de camadas de convolução e ativações.
        """
        return nn.Sequential(            
            nn.Conv2d(in_channels, out_channels, 3, padding=1),            
            nn.ReLU(inplace=True),            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def train_model(self, num_epochs, train_loader, learning_rate=0.001, weight_decay=1e-3):
        """
        Treina o modelo UNet com os dados de treinamento e calcula as métricas IoU e Dice.

        Args:
            num_epochs (int): Número de épocas de treinamento.
            train_loader (torch.utils.data.DataLoader): DataLoader com os dados de treinamento.
            learning_rate (float): Taxa de aprendizado (default: 0.001).
            weight_decay (float): Peso da regularização L2 (default: 1e-3).
        """

        optimizer = torch.optim.Adamax(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        self.to(self.device)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            total_iou = 0.0
            total_dice = 0.0

            for batch in tqdm(train_loader):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)

                labels = labels.unsqueeze(1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calcule a acurácia
                predicted = (outputs > 0.5).float()
                correct = (predicted == labels).sum().item()
                total_correct += correct
                total_samples += labels.numel()

                # Calcular a métrica IoU
                intersection = (predicted * labels).sum().item()
                union = (predicted + labels).sum().item() - intersection
                iou = intersection / union
                total_iou += iou

                # Calcular a métrica Dice
                dice = (2 * intersection) / (predicted.sum().item() + labels.sum().item())
                total_dice += dice

            average_loss = total_loss / len(train_loader)
            accuracy = total_correct / total_samples
            average_iou = total_iou / len(train_loader)
            average_dice = total_dice / len(train_loader)

            print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f} - IoU: {average_iou:.4f} - Dice: {average_dice:.4f}')
    
    def predict_image(self, image_path):
        """
        Predição de uma máscara.

        Args:
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Predicted mask as a tensor.
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
            'in_channels': self.in_channels,
            'num_filters': self.num_filters
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
            UNet: Instância do modelo carregado.
        """
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(n_classes=checkpoint['n_classes'], in_channels=checkpoint['in_channels'], num_filters=checkpoint['num_filters'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Model loaded from {filepath}')
        return model
