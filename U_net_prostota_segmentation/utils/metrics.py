# utils/metrics.py
import torch

def dice_coef(y_pred, y_true, smooth=1e-6):
    y_pred_flat, y_true_flat = y_pred.view(-1), y_true.view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()
    return (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)
