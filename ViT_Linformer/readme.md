# Introdução

Este repositório contém um projeto de classificação de imagens usando um modelo Vision Transformer (ViT) finetunado. O objetivo é classificar imagens em duas categorias: "Catarata ocular" e "Olho saudável". O projeto inclui código Python para treinar o modelo, avaliá-lo e realizar previsões em imagens individuais. Além disso, há uma aplicação web baseada em Flask que permite aos usuários fazer upload de imagens e obter previsões em tempo real.

## Conteúdo do Repositório

O projeto está organizado da seguinte forma:

1. **`ViTFineTuning.py`**: Este arquivo contém a implementação da classe `ViTFineTuning`, que é usada para treinar e avaliar o modelo ViT. Ele inclui funções para preparar o dataset, treinar o modelo, avaliar o modelo e fazer previsões em imagens individuais.

2. **`app.py`**: Este arquivo contém a aplicação web Flask que permite aos usuários fazer upload de imagens e obter previsões usando o modelo treinado.

3. **`modelos`**: Esta pasta contém os modelos pré-treinados salvos após o treinamento.

4. **`dados`**: Esta pasta deve conter o dataset de imagens organizado em subpastas, uma para cada classe. O dataset deve seguir a estrutura padrão do `torchvision.datasets.ImageFolder`.

5. **`static/uploads`**: Esta pasta é usada para armazenar temporariamente as imagens enviadas pelos usuários para classificação.

## Como Usar

### Treinamento do Modelo

Para treinar o modelo ViT, siga estas etapas:

1. Defina o caminho para o diretório do dataset no arquivo `ViTFineTuning.py`:

```python
dataset = "dados"
```

2. Crie uma instância da classe `ViTFineTuning` e treine o modelo executando o código de exemplo:

```python
cls = ViTFineTuning(dataset)
cls.train_model()
```

### Avaliação do Modelo

Você pode avaliar o desempenho do modelo treinado usando o dataset de validação. Execute o seguinte código de exemplo:

```python
cls.evaluate_model()
```

Isso imprimirá a acurácia na validação e mostrará uma matriz de confusão para avaliação visual.

### Previsões em Imagens Individuais

Para fazer previsões em imagens individuais, siga estas etapas:

1. Defina o caminho para o modelo pré-treinado e o caminho para a imagem de entrada no arquivo `app.py`:

```python
modelo_path = os.path.join('modelos', 'modelo_val_acc_x.xxxx.pt')
image_predict = os.path.join('image_teste', 'example_image.png')
```

2. Execute o aplicativo Flask:

```bash
python app.py
```

3. Abra um navegador da web e acesse `http://localhost:5000` para fazer upload de uma imagem e obter previsões em tempo real.

## Requisitos

Para executar o código neste repositório, você precisará das seguintes bibliotecas Python:

- PyTorch
- torchvision
- linformer
- vit-pytorch
- flask
- pillow
- tqdm
- numpy
- matplotlib
- seaborn
- scikit-learn

Você pode instalar as bibliotecas usando o gerenciador de pacotes `pip`:

```bash
pip install torch torchvision linformer vit-pytorch flask pillow tqdm numpy matplotlib seaborn scikit-learn
```

Além disso, é recomendável ter uma GPU disponível para treinamento mais rápido do modelo. Certifique-se de que o PyTorch esteja configurado para usar a GPU, se disponível.

## Créditos

O modelo Vision Transformer (ViT) usado neste projeto é baseado na implementação de [linformer](https://github.com/tatp22/linformer) e [vit-pytorch](https://github.com/lucidrains/vit-pytorch). Este projeto foi criado por [Seu Nome] e é fornecido sob a licença [MIT](LICENSE).