# Implementação do PixelCNN

Este repositório contém uma implementação em PyTorch do modelo PixelCNN, que é um modelo generativo utilizado para tarefas de geração de imagens. O PixelCNN é um tipo de modelo autoregressivo que gera imagens pixel por pixel, levando em consideração as dependências entre pixels. O modelo é projetado para capturar correlações locais em imagens e produzir amostras de imagem de alta qualidade e coesas.

## Arquitetura do Modelo

O modelo PixelCNN implementado neste repositório consiste nos seguintes componentes:

1. **Convolução com Máscara:**
   A classe `MaskedConvolution` estende o módulo `nn.Conv2d` e introduz o uso de máscaras para garantir a propriedade autoregressiva do PixelCNN. Ela aplica uma operação de convolução enquanto mascara os pixels futuros, assegurando que cada pixel dependa apenas dos pixels gerados anteriormente.

2. **Bloco Residual:**
   A classe `ResidualBlock` define um bloco residual utilizado no modelo. Ele é composto por camadas convolucionais com funções de ativação ReLU. As camadas de convolução com máscara dentro do bloco residual ajudam a capturar dependências espaciais na imagem.

3. **PixelCNN:**
   A classe `PixelCNN` representa o modelo PixelCNN principal. Ela é composta por uma camada de convolução de entrada, uma pilha de blocos residuais, camadas adicionais de convolução com máscara e uma camada de saída. O modelo gera imagens prevendo iterativamente o valor de cada pixel com base nos pixels gerados anteriormente.

## Uso

Para usar o modelo PixelCNN para geração de imagens, siga estes passos:

1. Importe os módulos necessários:

```python
import torch
import torch.nn as nn
```

2. Defina a classe `MaskedConvolution`, que estende `nn.Conv2d` e aplica máscaras para impor restrições autoregressivas.

3. Implemente a classe `ResidualBlock`, um bloco de construção para o modelo PixelCNN, que inclui convoluções com máscara e ativações ReLU.

4. Crie a classe `PixelCNN`, definindo a arquitetura principal do modelo com convoluções de entrada, blocos residuais, convoluções com máscara e uma camada de saída.

5. Instancie o modelo `PixelCNN` e utilize-o para gerar imagens.

Exemplo de uso:
```python
# Instancie o modelo PixelCNN
modelo_pixel_cnn = PixelCNN()

# Gere imagens usando o modelo
imagem_entrada = torch.randn(1, 1, 64, 64)  # Exemplo de tensor de imagem de entrada
imagem_gerada = modelo_pixel_cnn(imagem_entrada)
```

## Dependências

- PyTorch
- torch.nn
- torch.Tensor

## Reconhecimentos

Esta implementação é inspirada no modelo PixelCNN original proposto no artigo de pesquisa: "Pixel Recurrent Neural Networks", de Aaron van den Oord, et al.

## Autor

Felipe Meganha

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

Sinta-se à vontade para usar, modificar e distribuir este código para fins educacionais e de pesquisa. Se você considerar esta implementação útil, por favor, dê crédito referenciando este repositório.