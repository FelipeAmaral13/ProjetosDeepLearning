import matplotlib.pyplot as plt
import torch
import numpy as np

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def plot_sample(image, mask, prediction=None, save_path=None):
    image = denormalize(image)
    image = image.permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara Real')
    plt.axis('off')
    
    if prediction is not None:
        prediction = prediction.squeeze().numpy()
        plt.subplot(133)
        plt.imshow(prediction > 0.5, cmap='gray')
        plt.title('Predição')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, metrics_history, save_dir=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot de perdas
    plt.subplot(131)
    plt.plot(epochs, train_losses, label='Treino')
    plt.plot(epochs, val_losses, label='Validação')
    plt.title('Curvas de Perda')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    
    # Plot do Dice
    plt.subplot(132)
    plt.plot(epochs, [m['dice'] for m in metrics_history])
    plt.title('Coeficiente Dice')
    plt.xlabel('Época')
    plt.ylabel('Dice')
    
    # Plot do IoU
    plt.subplot(133)
    plt.plot(epochs, [m['iou'] for m in metrics_history])
    plt.title('IoU Score')
    plt.xlabel('Época')
    plt.ylabel('IoU')
    
    if save_dir:
        plt.savefig(f'{save_dir}/training_curves.png')
        plt.close()
    else:
        plt.show()