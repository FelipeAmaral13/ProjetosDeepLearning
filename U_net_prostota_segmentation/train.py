import torch
from torch.utils.data import DataLoader, TensorDataset
import config
from models.unet import UNetWithResNetBackbone
from data.data_loader import load_data
from utils.metrics import dice_coef
from utils.plotting import plot_learning_curve
import matplotlib.pyplot as plt

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    train_losses, val_losses, dice_scores = [], [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = sum([_train_step(model, X_batch, y_batch, optimizer, criterion) for X_batch, y_batch in train_loader])
        train_loss, val_loss, dice_score = running_loss / len(train_loader), *validate(model, valid_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        dice_scores.append(dice_score)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Dice Score: {dice_score}")
    plot_learning_curve(train_losses, val_losses, dice_scores)

def _train_step(model, X_batch, y_batch, optimizer, criterion):
    optimizer.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, valid_loader, criterion):
    model.eval()
    total_loss, dice_scores = 0.0, []
    with torch.no_grad():
        for X_val, y_val in valid_loader:
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val)
            total_loss += loss.item()
            dice_scores.append(dice_coef(y_pred > 0.5, y_val).item())
    return total_loss / len(valid_loader), sum(dice_scores) / len(dice_scores)

def predict(model, dataloader, threshold=0.5):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            y_pred = model(X_batch.to(config.DEVICE))
            preds.append((y_pred > threshold).float().cpu())
    return torch.cat(preds)

def plot_predictions(X, y_true, y_pred, num_samples=5):
    """Função para plotar exemplos das previsões."""
    indices = torch.randint(0, len(X), (num_samples,))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(X[idx, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Imagem Original")
        axes[i, 1].imshow(y_true[idx, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Máscara Real")
        axes[i, 2].imshow(y_pred[idx, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title("Predição")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Definir o caminho para salvar o modelo
    model_save_path = "unet_trained_model.pth"
    
    # Inicializar o modelo, otimizador e carregadores de dados
    model = UNetWithResNetBackbone().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    X_train, X_valid, y_train, y_valid = load_data(config.DATA_DIR)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Treinar o modelo
    train(model, train_loader, valid_loader, optimizer, criterion, config.NUM_EPOCHS)
    
    # Salvar o modelo treinado
    model.save_model(model_save_path)
    
    # Carregar o modelo (exemplo)
    model_loaded = UNetWithResNetBackbone().to(config.DEVICE)
    model_loaded.load_model(model_save_path, device=config.DEVICE)
    
    # Fazer predições nos dados de validação
    preds_val = predict(model_loaded, valid_loader)
    
    # Plotar as predições com as máscaras reais
    plot_predictions(X_valid, y_valid, preds_val, num_samples=5)
