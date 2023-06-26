# ChestRadioClass

Esta é uma classe que implementa um pipeline de classificação de radiografia torácica usando uma Rede Neural Convolucional (CNN). O pipeline inclui carregar e pré-processar as imagens de treinamento, realizar aumento de dados, construir e treinar o modelo CNN, avaliar o modelo e fazer previsões em novas imagens.

## Requirements

* numpy
* os
* pathlib
* glob
* cv2
* imblearn
* tqdm
* sklearn
* tensorflow
* keras
* matplotlib
* random

## Dados:

O dataset utilizado neste projeto é o `Tuberculosis (TB) Chest X-ray Database` (https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset). 