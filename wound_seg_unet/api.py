import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.unet import UNet
import cv2
from torch.serialization import add_safe_globals

# Adicionar numpy.scalar como um global seguro
add_safe_globals(['numpy._core.multiarray.scalar'])

app = FastAPI(title="Wound Segmentation API")

# Modelo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'checkpoints/best_model.pth'

try:
    # Primeira tentativa: carregar com weights_only=True
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except Exception as e:
    try:
        # Segunda tentativa: carregar com weights_only=False (se confiar na fonte do checkpoint)
        print("Tentando carregar o modelo com weights_only=False...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        raise

# Carregamento do modelo
model = UNet(n_channels=3, n_classes=1).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transformações para a imagem
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

class ImageRequest(BaseModel):
    image: str  # imagem em base64

class SegmentationResponse(BaseModel):
    mask: str  # máscara em base64
    confidence: float

def base64_to_image(base64_str: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def image_to_base64(image: np.ndarray) -> str:
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image")
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(request: ImageRequest):
    try:
        # Converter base64 para imagem
        image = base64_to_image(request.image)
        
        # Preparar imagem para o modelo
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inferência
        with torch.no_grad():
            prediction = model(image_tensor)
        
        # Processar predição
        mask = prediction.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Redimensionar máscara para tamanho original
        original_size = image.size[::-1]  # PIL Image size é (width, height)
        mask = cv2.resize(mask, original_size)
        
        # Calcular confiança média
        confidence = float(prediction.mean().cpu().numpy())
        
        # Converter máscara para base64
        mask_base64 = image_to_base64(mask)
        
        return SegmentationResponse(
            mask=mask_base64,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)