# Readme - Segmentação de Imagens com UNet e ResNeXtUNet

Este repositório contém implementações de duas arquiteturas de rede neural para segmentação de imagens: UNet e ResNeXtUNet. Essas redes são úteis em tarefas de segmentação semântica ou de instâncias, nas quais o objetivo é identificar objetos de interesse em uma imagem e criar máscaras que representem suas localizações.

## UNet

A arquitetura UNet é amplamente utilizada para tarefas de segmentação de imagens. A implementação aqui apresentada é uma versão simples da UNet e pode ser personalizada de acordo com o número de classes a serem segmentadas, o número de canais de entrada e o número de filtros iniciais. Os principais componentes da UNet incluem:

- **Encoder**: As camadas de convolução descendentes, que reduzem a resolução espacial da imagem.
- **Decoder**: As camadas de convolução ascendentes, que aumentam a resolução espacial e combinam informações de diferentes escalas.
- **Camada de saída**: Uma camada de convolução final que produz as máscaras segmentadas.

## ResNeXtUNet

A ResNeXtUNet é uma variação da UNet que utiliza uma arquitetura ResNet como backbone. Isso permite que a rede aproveite os benefícios de pré-treinamento em tarefas de classificação de imagem, melhorando o desempenho na segmentação de objetos. A arquitetura da ResNeXtUNet inclui:

- **Backbone ResNet**: Uma rede ResNet pré-treinada que extrai características das imagens de entrada.
- **Codificadores**: Camadas de convolução que capturam informações de diferentes escalas.
- **Decodificadores**: Camadas de convolução ascendentes que combinam informações de diferentes escalas.
- **Camada de saída**: Uma camada de convolução final que produz as máscaras segmentadas.

## Como Usar

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

- PyTorch
- torchvision
- PIL (Pillow)
- tqdm
- numpy
- matplotlib
- albumentations

### Treinamento

Para treinar os modelos, siga os seguintes passos:

1. Organize suas imagens e máscaras em diretórios separados, onde as máscaras correspondentes tenham nomes semelhantes aos das imagens.

2. Configure os parâmetros de treinamento, como o diretório de imagem, diretório de máscaras, tamanho do lote (batch size) e transformações de dados desejadas.

3. Use as funções `get_loaders` e `train_model` para carregar os dados e iniciar o treinamento.

4. A função `train_model` treina o modelo por um número especificado de épocas, calculando métricas como perda (loss), acurácia, IoU (Intersection over Union) e Dice.

### Inferência

Após treinar o modelo ou carregar um modelo treinado, você pode realizar inferências em imagens individuais usando a função `predict_image`. Basta fornecer o caminho para a imagem de entrada, e o modelo irá gerar uma máscara prevista.

## Exemplo de Uso

O código de exemplo fornece um fluxo completo de treinamento e inferência de um modelo ResNeXtUNet. Ele inclui:

- Carregamento de dados usando a classe `CustomDataset` para criar conjuntos de treinamento e teste.
- Configuração de transformações de dados para treinamento e inferência.
- Treinamento do modelo por um número especificado de épocas.
- Realização de inferência em uma imagem de exemplo e exibição da máscara prevista.

Você pode personalizar esse código de acordo com seus próprios dados e requisitos de treinamento.

## Referências

- UNet: https://arxiv.org/abs/1505.04597
- ResNet: https://arxiv.org/abs/1512.03385
- ResNeXt: https://arxiv.org/abs/1611.05431
- Albumentations: https://albumentations.ai/

