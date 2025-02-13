import requests
import base64
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    return image

def segment_image(image_path, api_url="http://localhost:8000/segment"):
    # Carregar e codificar imagem
    image_base64 = load_image_base64(image_path)
    
    # Preparar requisição
    payload = {
        "image": image_base64
    }
    
    # Fazer requisição para a API
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Converter máscara de base64 para imagem
        mask = base64_to_image(result['mask'])
        confidence = result['confidence']
        
        return mask, confidence
    
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None, None

def visualize_results(image_path, mask, confidence):
    # Carregar imagem original
    original = np.array(Image.open(image_path))
    
    # Criar visualização
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara Segmentada')
    plt.axis('off')
    
    # Sobrepor máscara na imagem original
    overlay = original.copy()
    mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    overlay = cv2.addWeighted(overlay, 0.2, mask_rgb, 0.3, 0)
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title(f'Sobreposição\nConfiança: {confidence:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Exemplo de uso
    image_path = r"dataset\test\images\1443.png"
    
    print("Enviando imagem para segmentação...")
    mask, confidence = segment_image(image_path)
    
    if mask is not None:
        print(f"Segmentação concluída com confiança: {confidence:.2f}")
        visualize_results(image_path, mask, confidence)
    else:
        print("Falha na segmentação")

if __name__ == "__main__":
    main()