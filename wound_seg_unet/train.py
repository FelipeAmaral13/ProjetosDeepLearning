import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models.unet import UNet
from utils.data_loader import get_dataloaders
from utils.metrics import MetricTracker
from utils.visualization import plot_sample, plot_training_curves

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    metrics_history = {'train': [], 'val': []}
    best_dice = 0
    
    for epoch in range(num_epochs):
        # Modo treino
        model.train()
        epoch_loss = 0
        train_metric_tracker = MetricTracker()
        
        train_bar = tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} [Treino]')
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_metric_tracker.update(outputs.detach().cpu(), masks.cpu())
            
            # Atualizar métricas na barra de progresso
            train_metrics = train_metric_tracker.get_metrics()
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{train_metrics["dice"]:.4f}',
                'iou': f'{train_metrics["iou"]:.4f}'
            })
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        final_train_metrics = train_metric_tracker.get_metrics()
        metrics_history['train'].append(final_train_metrics)
        
        # Modo validação
        model.eval()
        val_loss = 0
        val_metric_tracker = MetricTracker()
        
        val_bar = tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                val_metric_tracker.update(outputs.cpu(), masks.cpu())
                
                # Atualizar métricas na barra de progresso
                val_metrics = val_metric_tracker.get_metrics()
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{val_metrics["dice"]:.4f}',
                    'iou': f'{val_metrics["iou"]:.4f}'
                })
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        final_val_metrics = val_metric_tracker.get_metrics()
        metrics_history['val'].append(final_val_metrics)
        
        # Imprimir resumo da época
        print(f'\nResumo da Época {epoch+1}:')
        print(f'Treino - Loss: {train_loss:.4f}, Dice: {final_train_metrics["dice"]:.4f}, IoU: {final_train_metrics["iou"]:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice: {final_val_metrics["dice"]:.4f}, IoU: {final_val_metrics["iou"]:.4f}')
        
        # Salvar melhor modelo baseado no Dice de validação
        if final_val_metrics['dice'] > best_dice:
            best_dice = final_val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'train_metrics': final_train_metrics,
                'val_metrics': final_val_metrics,
            }, f'{save_dir}/best_model.pth')
            print(f'Melhor modelo salvo com Dice: {best_dice:.4f}')
        
        # Plotar e salvar exemplos a cada 5 épocas
        if (epoch + 1) % 5 == 0:
            images, masks = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                predictions = model(images[:4].to(device))
            
            for i in range(4):
                plot_sample(
                    images[i], masks[i], predictions[i].cpu(),
                    save_path=f'{save_dir}/sample_epoch_{epoch+1}_img_{i}.png'
                )
    
    # Plotar curvas de treinamento
    plot_training_curves(
        train_losses, 
        val_losses, 
        metrics_history['val'],  # Usando métricas de validação para os gráficos
        save_dir
    )
    return train_losses, val_losses, metrics_history

def main():
    # Configurações
    BATCH_SIZE = 12
    IMG_SIZE = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar dados
    train_loader, val_loader = get_dataloaders(
        'dataset',
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )
    
    # Inicializar modelo
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    
    # Otimizador e função de perda
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Treinar modelo
    train_losses, val_losses, metrics_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )

if __name__ == '__main__':
    main()