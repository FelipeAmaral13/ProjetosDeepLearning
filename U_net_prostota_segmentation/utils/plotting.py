# utils/plotting.py
import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, val_losses, dice_scores):
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(dice_scores, label="Dice Score")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.show()
