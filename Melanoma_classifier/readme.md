Melanoma Classification using Convolutional Neural Networks

Este projeto é um exemplo de classificação de melanoma em maligno ou benigno usando redes neurais convolucionais (CNN).
Dados

O dataset utilizado neste projeto é o melanoma_cancer_dataset (https://www.kaggle.com/search?q=melanoma-skin-cancer-dataset-of-10000-images), que contém imagens de melanomas de pacientes. Este dataset é composto por um conjunto de treinamento e um conjunto de teste. As imagens foram pré-processadas para terem tamanho 32x32.
Dependências

Para executar este projeto, você precisará ter as seguintes bibliotecas Python instaladas:

numpy
os
cv2
random
matplotlib
tqdm
tensorflow
keras
sklearn

Instalação

Use o comando abaixo para instalar todas as dependências do projeto:


pip install -r requirements.txt

Como funciona
Classe MelanomaClass

A classe MelanomaClass é responsável por carregar os dados, treinar o modelo, avaliar o modelo, salvar o modelo e realizar previsões.

load_data(): carrega os dados de treinamento e teste, redimensiona as imagens e retorna um tupla com as imagens e suas respectivas classes.
plot_sample_images(): plota duas imagens de exemplo, uma benigna e uma maligna, para visualização.
train(batch_size=2, epochs=20, validation_split=0.2): treina o modelo com os dados carregados e retorna um histórico de treinamento.
evaluate(): avalia o modelo com os dados de teste e retorna a perda.
save_model(): salva o modelo treinado em um arquivo .h5.
load_model(): carrega o modelo salvo no arquivo .h5.
predict(image_path): faz uma previsão da classe da imagem no caminho especificado.