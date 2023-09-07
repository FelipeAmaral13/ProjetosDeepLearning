from flask import Flask, request, render_template
import os
from PIL import Image
import torch
from linformer import Linformer
from vit_pytorch.efficient import ViT
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

app = Flask(__name__)

def create_model():
    """
    Cria um modelo Vision Transformer (ViT) para classificação de imagens.

    Retorna:
        model (torch.nn.Module): O modelo ViT.
        device (torch.device): O dispositivo no qual o modelo está carregado.
    """
    in_features = 128

    # Define a camada Linformer para o transformador
    fine_tuning_transformer_layer = Linformer(
        dim=in_features,
        seq_len=49+1,
        depth=12,
        heads=8,
        k=64
    )
    
    # Cria o modelo ViT
    model = ViT(
        dim=in_features,
        image_size=224,
        patch_size=32,
        num_classes=2,
        transformer=fine_tuning_transformer_layer,
        channels=3
    )

    # Substitui a camada de classificação do modelo para classificação binária
    model.head = nn.Linear(in_features, 2)
    
    # Determina o dispositivo para treinamento do modelo (GPU se disponível, caso contrário, CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define otimizador, agendador de taxa de aprendizado e critério de perda
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    return model, device

# Define o caminho para o modelo pré-treinado
modelo_path = os.path.join('modelos', 'modelo_val_acc_0.8729.pt')

# Carrega o modelo pré-treinado e cria as instâncias de modelo e dispositivo
state_dict = torch.load(modelo_path)
modelo, device = create_model()
modelo.load_state_dict(state_dict)
modelo.eval()

# Define a pasta de upload para imagens
UPLOAD_FOLDER = r'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Manipula o upload e classificação de imagens.

    Retorna:
        HTML renderizado com imagem e rótulo.
    """
    image_path = None
    label = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "Nenhuma imagem encontrada. Por favor, selecione uma imagem e tente novamente."

        file = request.files['file']

        if file.filename == '':
            return "Nome do arquivo vazio. Por favor, selecione uma imagem e tente novamente."

        if file:
            # Salva a imagem enviada no servidor
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            image = Image.open(file_path)

            # Aplica transformações na imagem
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = transform(image).unsqueeze(0)
            image = image.to(device)

            # Realiza inferência usando o modelo
            with torch.no_grad():
                outputs = modelo(image)
                _, predicted = outputs.max(1)

            # Determina o rótulo com base na predição do modelo
            label = "Catarata ocular" if predicted == 0 else "Olho saudável"
            image_path = os.path.join('static', 'uploads', file.filename)

            print(f"Caminho do arquivo de imagem: {image_path}")

    return render_template('index.html', image_path=image_path, label=label)

if __name__ == '__main__':
    # Garante que a pasta de upload exista
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Executa a aplicação Flask
    app.run(debug=True, host='0.0.0.0', port='5000')
