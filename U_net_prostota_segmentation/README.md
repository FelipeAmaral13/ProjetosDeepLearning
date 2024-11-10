
# Projeto U-Net para Segmentação de Imagens de Próstata

Este projeto implementa uma arquitetura de rede neural U-Net com backbone ResNet para segmentação de imagens de próstata, utilizando PyTorch. A aplicação do modelo visa a segmentação precisa de imagens médicas para auxiliar na análise de imagens de próstata e máscaras associadas.

## Estrutura do Projeto

```
project_root/
├── models/
│   ├── __init__.py
│   ├── unet.py               # Define a arquitetura do modelo U-Net
│   ├── conv_block.py         # Define o bloco convolucional reutilizável
├── data/
│   ├── __init__.py
│   ├── data_loader.py        # Funções e classes para carregar e processar os dados
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # Funções para cálculo de métricas, como coeficiente de Dice
│   ├── plotting.py           # Funções para visualização dos dados e métricas de treinamento
├── train.py                  # Script principal para treinar e avaliar o modelo
├── config.py                 # Configurações gerais do projeto (hiperparâmetros, paths)
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação do projeto
```

## Requisitos

Para instalar as dependências do projeto, execute:

```bash
pip install -r requirements.txt
```

## Configurações

O arquivo `config.py` contém as principais configurações do projeto, como:

- `DEVICE`: dispositivo para treinamento (`cuda` ou `cpu`)
- `DATA_DIR`: caminho para o diretório de dados
- `IM_HEIGHT` e `IM_WIDTH`: dimensões das imagens
- `BATCH_SIZE`: tamanho do batch
- `LEARNING_RATE`: taxa de aprendizado
- `NUM_EPOCHS`: número de épocas de treinamento

## Executando o Treinamento

Para treinar o modelo, utilize o script `train.py`:

```bash
python train.py
```

Durante o treinamento, o script:

1. Carrega os dados de imagem e máscara a partir do diretório especificado.
2. Treina a U-Net usando uma função de perda binária e o otimizador Adam.
3. Calcula e exibe métricas como a Loss e o Dice Score por época.
4. Gera um gráfico das métricas de aprendizado utilizando `utils/plotting.py`.
5. Salva o modelo treinado no caminho especificado.

## Funções Importantes

- `train.py`: script principal que realiza o treinamento, validação e salvamento do modelo treinado.
- `models/unet.py`: define a arquitetura U-Net com backbone ResNet para melhor desempenho na segmentação.
- `data/data_loader.py`: realiza o carregamento e pré-processamento das imagens de entrada e máscaras para treinamento.
- `utils/metrics.py`: inclui a função `dice_coef` para calcular a similaridade entre as máscaras preditas e reais.
- `utils/plotting.py`: funções para visualização do processo de treinamento, como curvas de aprendizado.

## Exemplo de Uso

Para realizar predições, utilize a função `predict()` disponível no script `train.py`, que permite realizar inferência nos dados de validação ou teste, utilizando um modelo previamente treinado.

## Visualizando Resultados

A função `plot_predictions()` permite visualizar amostras de imagens com as máscaras reais e preditas pelo modelo, facilitando a avaliação visual dos resultados.

## Contribuição

Contribuições são bem-vindas. Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto é licenciado sob a MIT License.
