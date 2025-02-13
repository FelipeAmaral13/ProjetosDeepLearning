import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

class MetricTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        
    def update(self, pred, target):
        pred = (pred > 0.5).float()
        self.dice_scores.append(dice_coefficient(pred, target).item())
        self.iou_scores.append(iou_score(pred, target).item())
    
    def get_metrics(self):
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores)
        }