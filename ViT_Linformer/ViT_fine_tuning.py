# Imports
import os
import copy
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from linformer import Linformer
from vit_pytorch.efficient import ViT
from torchvision import datasets, transforms
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ViTFineTuning:
    def __init__(self, dataset):
        """
        Inicializa a classe ViTFineTuning.

        Args:
            dataset (str): O caminho para o diretório do dataset de imagens.
        """
        self.dataset_path = datasets.ImageFolder(dataset)  # Carrega o dataset
        self.class_names = self.dataset_path.classes  # Obtém os nomes das classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Verifica o dispositivo disponível

        self.dataloaders = self.dataset('dados')  # Cria dataloaders para treino e validação
        self.modelo = self.model()  # Inicializa o modelo
        self.exp_lr_scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)  # Inicializa o agendador de taxa de aprendizado

    def dataset(self):
        """
        Prepara o dataset, divide em treino e validação, e cria dataloaders.

        Args:
            dataset (str): O caminho para o diretório do dataset.

        Returns:
            dataloaders (dict): Um dicionário com dataloaders para treino e validação.
        """
        # Parâmetros para dataloaders
        batch_size = 32
        len_dataset = len(self.dataset_path)
        len_train = int(0.7*len_dataset)
        len_val = len_dataset - len_train
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset_path, [len_train, len_val])

        # Tamanhos dos datasets de treino e validação
        self.dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        # Transformações de dados
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        train_dataset.dataset.transform = data_transforms['train']

        # Cria dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4
        )

        dataloaders = {'train': self.train_dataloader, 'val': self.val_dataloader}

        return dataloaders

    def plot_samples(self):
        """
        Plota amostras do dataset de treino.
        """
        inputs, labels = next(iter(self.train_dataloader))
        inputs = inputs.numpy()
        labels = labels.numpy()
        label_names = [self.class_names[label] for label in labels]

        fig, axes = plt.subplots(figsize = (12, 8), nrows = 4, ncols = 8)
        axes = axes.flatten()

        for i, (image, label) in enumerate(zip(inputs, label_names)):
            image = np.transpose(image, (1, 2, 0))
            image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            image = np.clip(image, 0, 1)
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(label)

        plt.tight_layout()
        plt.show()

    def model(self):
        """
        Cria o modelo ViT.

        Returns:
            modelo (torch.nn.Module): O modelo ViT.
        """
        in_features = 128
        fine_tuning_transformer_layer = Linformer(
            dim=in_features,
            seq_len=49+1,
            depth=12,
            heads=8,
            k=64)

        modelo = ViT(dim = in_features,
             image_size = 224,
             patch_size = 32,
             num_classes = 2,
             transformer = fine_tuning_transformer_layer,
             channels = 3)

        modelo.head = nn.Linear(in_features, len(self.class_names))
        modelo = modelo.to(self.device)

        self.optimizer = optim.Adam(modelo.parameters(), lr=3e-5)
        self.exp_lr_scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        return modelo

    def train_model(self):
        """
        Treina o modelo ViT.
        """
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        num_epochs = 10
        best_model_wts = copy.deepcopy(self.modelo.state_dict())
        best_acc = 0.0
        prev_best_acc = 0.0

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.modelo.train()
                else:
                    self.modelo.eval()

                running_loss = 0.0
                running_corrects = 0

                dataloader = self.dataloaders[phase]

                with torch.set_grad_enabled(phase == 'train'):
                    data_loader = tqdm(dataloader, total=len(dataloader), unit="batch")

                    for batch_idx, (inputs, labels) in enumerate(data_loader):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        self.optimizer.zero_grad()

                        outputs = self.modelo(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        data_loader.set_postfix(loss=loss.item())

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = (running_corrects.double() / self.dataset_sizes[phase]).item()

                    if phase == 'train':
                        train_loss_history.append(epoch_loss)
                        train_acc_history.append(epoch_acc)
                    else:
                        val_loss_history.append(epoch_loss)
                        val_acc_history.append(epoch_acc)

                    print(f"Epoch {epoch}/{num_epochs - 1} --> Phase: {phase} Error: {epoch_loss} Acc: {epoch_acc}")

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.modelo.state_dict())

                    if phase == 'train':
                        self.exp_lr_scheduler.step()

                print(f"\nMelhor Acurácia em Validação: {best_acc}")

                if best_acc > prev_best_acc:
                    self.save_model(best_acc)
                    prev_best_acc = best_acc

        self.plot_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    def plot_training_history(self, train_loss, val_loss, train_acc, val_acc):
        """
        Plota os históricos de treinamento (erro e acurácia) em um gráfico.

        Args:
            train_loss (list): Lista com valores de erro de treinamento.
            val_loss (list): Lista com valores de erro de validação.
            train_acc (list): Lista com valores de acurácia de treinamento.
            val_acc (list): Lista com valores de acurácia de validação.
        """
        epoch = range(1, len(train_loss) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epoch, train_loss, label='Erro em Treino')
        ax[0].plot(epoch, val_loss, label='Erro em Validação')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Erro')
        ax[0].legend()

        ax[1].plot(epoch, train_acc, label='Acurácia em Treino')
        ax[1].plot(epoch, val_acc, label='Acurácia em Validação')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Acurácia')
        ax[1].legend()

        plt.show()

    def evaluate_model(self):
        """
        Avalia o modelo usando o dataset de validação.
        """
        self.modelo.eval()
        running_corrects = 0
        all_labels = []
        all_preds = []

        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.modelo(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        accuracy = running_corrects.double() / self.dataset_sizes['val']
        print('Acurácia na Validação: {:.4f}'.format(accuracy.item()))

        confusion = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.show()

    def save_model(self, val_accuracy):
        """
        Salva o modelo treinado com base na acurácia de validação.

        Args:
            val_accuracy (float): Acurácia de validação do modelo.
        """
        if self.modelo is not None:
            torch.save(self.modelo.state_dict(), os.path.join('modelos', f'modelo_val_acc_{val_accuracy:.4f}.pt'))
        else:
            print("Modelo não criado. Chame create_model() primeiro.")

    def load_model(self, modelo):
        """
        Carrega um modelo pré-treinado.

        Args:
            modelo (str): Caminho para o modelo pré-treinado.

        Returns:
            modelo (torch.nn.Module): O modelo carregado.
        """
        if self.modelo is not None:
            self.modelo.load_state_dict(torch.load(modelo))
            return self.modelo
        else:
            print("Modelo não criado. Chame create_model() primeiro.")
            return None

    def predict_image(self, image_path, modelo):
        """
        Realiza previsões para uma imagem usando o modelo carregado.

        Args:
            image_path (str): Caminho para a imagem de entrada.
            modelo (str): Caminho para o modelo pré-treinado.

        Returns:
            predicted (int): Classe prevista para a imagem.
        """
        modelo = self.load_model(modelo)
        if modelo is not None:
            modelo.eval()
            modelo.to(self.device)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)
            image = image.to(self.device)

            with torch.no_grad():
                outputs = modelo(image)
                _, predicted = outputs.max(1)

            return predicted
        else:
            return None

# Exemplo de uso da classe ViTFineTuning
# dataset = "dados"
# cls = ViTFineTuning(dataset)
# cls.train_model()
# cls.evaluate_model()

# Exemplo de previsão de imagem
# image_predict = r"path_image"
# modelo = r"path_model.pt"
# cls.predict_image(image_predict, modelo)
