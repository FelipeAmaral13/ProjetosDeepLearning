import numpy as np
import os
from pathlib import Path
from glob import glob
import cv2
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras import  layers, models, Sequential
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import random

class ChestRadioClass: 

    def __init__(self) -> None:
        """
        Inicializa uma instância da classe ChestRadioClass.

        Atributos:
        - train_data (list): Lista vazia para armazenar os dados de treinamento das imagens.
        - train_labels (list): Lista vazia para armazenar os rótulos de treinamento das imagens.
        - image_dim (int): Dimensão das imagens (largura e altura).
        - image_channel (int): Número de canais de cor das imagens (RGB).
        """
        self.train_data = []
        self.train_labels = []
        self.image_dim = 32
        self.image_channel = 3
       
    def load_images(self, path_images):
        """
        Carrega as imagens de um diretório.

        Args:
        - path_images (str): Caminho do diretório contendo as imagens.

        Returns:
        - images (list): Lista contendo os caminhos completos das imagens carregadas.
        """
        images = glob(os.path.join(path_images, '*.png'))
        return images        

    
    def train_dataset(self, path_case_label, case_label):
        """
        Carrega e pré-processa as imagens de treinamento.

        Args:
        - path_case_label (str): Caminho do diretório contendo as imagens de treinamento.
        - case_label: Rótulo do caso.

        Returns:
        - train_data (np.array): Array contendo os dados de treinamento das imagens pré-processadas.
        - train_labels (np.array): Array contendo os rótulos de treinamento das imagens.
        """
        images = self.load_images(path_case_label)     

        for img in tqdm(images):
            img = cv2.imread(str(img))
            img = cv2.resize(img, (self.image_dim, self.image_dim))
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=np.array(img)
            img = img / 255
            label = case_label
            self.train_data.append(img)
            self.train_labels.append(label)

        train_data = np.array(self.train_data)
        train_labels = np.array(self.train_labels)

        return train_data, train_labels
    
    def create_samples(self, train_data_case, train_data_label):
        """
        Realiza a geração de amostras sintéticas usando a técnica SMOTE.

        Args:
        - train_data_case (np.array): Array contendo os dados de treinamento do caso.
        - train_data_label (np.array): Array contendo os rótulos de treinamento do caso.

        Returns:
        - train_data_final (np.array): Array contendo os dados de treinamento finais, incluindo as amostras sintéticas.
        - train_labels_final (np.array): Array contendo os rótulos de treinamento finais, incluindo os rótulos das amostras sintéticas.
        """
        smt = SMOTE()
        train_rows = len(train_data_case)
        train_data_case = train_data_case.reshape(train_rows, -1)
        train_data_final, train_labels_final = smt.fit_resample(train_data_case, train_data_label)

        label_encoder = LabelEncoder()
        train_labels_final = label_encoder.fit_transform(train_labels_final)

        return train_data_final, train_labels_final

    def plot_random_images(self, train_data_final, train_labels_final):
        """
        Plota imagens aleatórias do conjunto de dados SMOTE.

        Args:
        - train_data_final (np.array): Array contendo os dados de treinamento finais.
        - train_labels_final (np.array): Array contendo os rótulos de treinamento finais.
        """
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
        fig.suptitle("Random Images from SMOTE Dataset", fontsize=16)
        axes = axes.flatten()

        unique_labels = np.unique(train_labels_final)
        for i, ax in enumerate(axes):
            label = random.choice(unique_labels)
            indices = np.where(train_labels_final == label)[0]
            random_index = random.choice(indices)
            image = train_data_final[random_index].reshape(self.image_dim, self.image_dim, self.image_channel)

            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    
    def data_augmentation(self):
        """
        Realiza a data augmentation das imagens.

        Returns:
        - data_aug (Sequential): Modelo sequencial contendo as transformações de data augmentation.
        """
        data_aug = Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.image_dim, self.image_dim, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)                
            ])

        return data_aug

    def cnn_model(self):
        """
        Constrói o modelo de rede neural convolucional (CNN).

        Returns:
        - model (Sequential): Modelo sequencial da CNN.
        """
        data_aug = self.data_augmentation()      

        model = models.Sequential([
            data_aug,
            layers.Conv2D(self.image_dim, (3, 3), activation='relu', input_shape=(self.image_dim, self.image_dim, self.image_channel)) ,
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax'),
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
            )
    
        return model

    def train_model(self, train_data_final, train_labels_final):
        """
        Treina o modelo de CNN usando os dados de treinamento.

        Args:
        - train_data_final (np.array): Array contendo os dados de treinamento finais.
        - train_labels_final (np.array): Array contendo os rótulos de treinamento finais.

        Returns:
        - history (History): Objeto contendo o histórico do treinamento do modelo.
        """
        self.model = self.cnn_model()
        train_data2 = train_data_final.reshape(-1, self.image_dim, self.image_dim, self.image_channel)

        X_train, X_test, y_train, y_test = train_test_split(
            train_data2,
            train_labels_final,
            test_size=0.2,
            )
        
        reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', 
                              factor = 0.5, 
                              patience = 2, 
                              verbose = 1, 
                              mode = 'max', 
                              min_lr = 0.00001)
    
        callbacks_list = [reduce_lr]

        history = self.model.fit(
            np.array(X_train),
            np.array(y_train),
            epochs=10,
            validation_data=(np.array(X_test), np.array(y_test)),
            verbose = 1,
            callbacks = callbacks_list
            )
        
        self.save_model(self.model)
        accuracy, report, matrix = self.evaluate_model(X_test, y_test)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        return history
    
    def plot_training_history(self, history):
        """
        Plota as curvas de perda e acurácia durante o treinamento do modelo.

        Args:
        - history (History): Objeto contendo o histórico do treinamento do modelo.
        """
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    
    def save_model(self, model):
        """
        Salva o modelo treinado em um arquivo.

        Args:
        - model: Modelo treinado a ser salvo.
        """
        if model is not None:
            self.model.save(os.path.join("Cap7", "modelos", "cnn_model.h5"), save_format="tf")
        else:
            raise ValueError("No model trained yet. Please train the model first.")
    
    def load_model(self, model_path):
        """
        Carrega um modelo salvo a partir de um arquivo.

        Args:
        - model_path (str): Caminho do arquivo contendo o modelo salvo.

        Returns:
        - model: Modelo carregado.
        """
        model = tf.keras.models.load_model(model_path)
        return model
    
    def evaluate_model(self, test_data, test_labels):
        """
        Avalia o modelo usando os dados de teste.

        Args:
        - test_data (np.array): Array contendo os dados de teste.
        - test_labels (np.array): Array contendo os rótulos de teste.

        Returns:
        - accuracy (float): Acurácia do modelo.
        - report (str): Relatório de classificação.
        - matrix (np.array): Matriz de confusão.
        """
        predictions = self.model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(test_labels, predicted_labels)
        report = classification_report(test_labels, predicted_labels)
        matrix = confusion_matrix(test_labels, predicted_labels)
        
        return accuracy, report, matrix


    def predict_image(self, image_path, model):
        """
        Faz a previsão de um único exemplo de imagem usando o modelo treinado.

        Args:
        - image_path (str): Caminho da imagem a ser prevista.
        - model: Modelo treinado a ser usado para fazer a previsão.

        Returns:
        - predicted_label (int): Rótulo previsto para a imagem.
        """
        test_data = []
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (self.image_dim, self.image_dim))
        if img.shape[2] == 1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img / 255
        test_data.append(img)

        test_data1 = np.array(test_data)
        predictions = model.predict(np.array(test_data1))
        predicted_label = np.argmax(predictions)

        return predicted_label



if __name__ == "__main__":
    data_dir = Path(os.path.join('TB_Chest_Radiography_Database'))
    normal_cases_dir = os.path.join(data_dir, 'Normal')
    Tuberculosis_cases_dir = os.path.join(data_dir, 'Tuberculosis')

    cls = ChestRadioClass()
    train_data_normal, train_labels_normal = cls.train_dataset(normal_cases_dir, 'Normal')
    train_data_tuberculis, train_labels_tuberculis = cls.train_dataset(Tuberculosis_cases_dir, 'Tuberculosis')

    train_data_final, train_labels_final = cls.create_samples(train_data_tuberculis, train_labels_tuberculis)
    cls.plot_random_images(train_data_final, train_labels_final)

    history = cls.train_model(train_data_final, train_labels_final)
    cls.plot_training_history(history)

    model_load = cls.load_model(os.path.join("Cap7", "modelos", "cnn_model.h5"))

    image_test = r'TB_Chest_Radiography_Database\Normal\Normal-2902.png'
    predicted_label = cls.predict_image(image_test, model_load)
