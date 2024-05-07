# Detecção de Objetos com Telegram Bot

Este é um aplicativo simples para detectar objetos em imagens enviadas pelo Telegram. O aplicativo utiliza a biblioteca YOLO (You Only Look Once) para a detecção de objetos e o Telegram Bot API para comunicação.

## Pré-requisitos

Certifique-se de ter as seguintes bibliotecas Python instaladas:

- `telepot`
- `magic`
- `opencv-python`
- `ultralytics`

Além disso, você precisará do modelo YOLO pré-treinado (por exemplo, `yolov8s.pt`) disponível na pasta `model`.

## Instalação

1. Clone este repositório em sua máquina local:

```
git clone https://github.com/seu-usuario/nome-do-repositorio.git
```

2. Navegue até o diretório do projeto:

```
cd nome-do-repositorio
```

3. Instale as dependências utilizando o `pip`:

```
pip install -r requirements.txt
```

## Configuração

Antes de executar o aplicativo, você precisa obter um token de acesso do Telegram Bot API. Siga estas etapas:

1. Inicie uma conversa com o [BotFather](https://t.me/botfather) no Telegram.
2. Use o comando `/newbot` para criar um novo bot e siga as instruções.
3. Após a criação do bot, o BotFather fornecerá um token de acesso. Copie esse token.

## Uso

Execute o aplicativo Python fornecendo o token do bot como argumento de linha de comando:

```
python main.py SEU_TOKEN_DO_BOT
```

Agora, você pode enviar imagens para o bot no Telegram e ele detectará os objetos nelas. O bot enviará de volta a imagem com caixas delimitadoras e rótulos dos objetos detectados.

## Funcionamento do Código

O código consiste em dois arquivos principais:

- `main.py`: Este é o arquivo principal que lida com a comunicação do bot Telegram e o processamento das imagens recebidas.
- `yolo_predict.py`: Este arquivo contém a classe `ObjectDetector`, que encapsula a lógica de detecção de objetos usando o modelo YOLO.

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema ou quiser melhorar este aplicativo, sinta-se à vontade para abrir uma [issue](https://github.com/seu-usuario/nome-do-repositorio/issues) ou enviar um [pull request](https://github.com/seu-usuario/nome-do-repositorio/pulls).

## Licença

Este projeto é licenciado sob a [MIT License](LICENSE).