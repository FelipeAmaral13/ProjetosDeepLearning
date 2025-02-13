# Wound Segmentation API

Este projeto implementa um modelo de segmentação de feridas utilizando a arquitetura U-Net. O sistema inclui:

- Uma API para segmentação de imagens de feridas com FastAPI.
- Um cliente para envio de imagens e visualização dos resultados.
- Código para treinamento do modelo U-Net.
- Funções auxiliares para métricas, carregamento de dados e visualização.

## Estrutura do Projeto

```
├── api.py              # API FastAPI para inferência
├── cliente.py          # Cliente para consumir a API
├── train.py            # Treinamento do modelo U-Net
├── utils
│   ├── data_loader.py      # Carregamento dos dados
│   ├── metrics.py          # Cálculo de métricas (Dice, IoU)
│   ├── visualization.py    # Funções para visualização de imagens e métricas
├── models
│   ├── unet.py             # Implementação do modelo U-Net
├── requirements.txt    # Dependências do projeto
└── checkpoints/        # Modelos treinados
```

## Requisitos

- Python 3.8+
- PyTorch
- FastAPI
- OpenCV
- PIL
- Matplotlib
- Torchvision
- Requests

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/FelipeAmaral13/wound-segmentation.git
   cd wound-segmentation
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

### Executando a API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

A API terá os seguintes endpoints:

- `POST /segment` – Recebe uma imagem em base64 e retorna a máscara segmentada.
- `GET /health` – Verifica se a API está funcionando corretamente.

### Testando a API com o Cliente

```bash
python cliente.py
```

O script `cliente.py` carrega uma imagem, envia para a API e exibe os resultados.

### Treinamento do Modelo

```bash
python train.py
```

O modelo U-Net será treinado e salvo na pasta `checkpoints/`.

Métricas de avaliação do treinamento do modelo:

![Image](https://github.com/user-attachments/assets/0eb4f12a-e030-4e5e-8b40-12500d9b43f4)

Exemplo de segmentação na inferência do modelo:

![Image](https://github.com/user-attachments/assets/581d6c81-798f-41de-8bed-16c905fdc9f0)

