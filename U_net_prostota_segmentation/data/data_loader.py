# data/data_loader.py
import os
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import config

def load_data(data_dir):
    image_ids = next(os.walk(os.path.join(data_dir, "images")))[2]
    X, y = np.zeros((len(image_ids), config.IM_HEIGHT, config.IM_WIDTH, 1), dtype=np.float32), np.zeros((len(image_ids), config.IM_HEIGHT, config.IM_WIDTH, 1), dtype=np.float32)
    for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
        X[n] = np.array(Image.open(os.path.join(data_dir, "images", id_)).convert('L').resize((config.IM_WIDTH, config.IM_HEIGHT))).reshape((config.IM_HEIGHT, config.IM_WIDTH, 1)) / 255.0
        y[n] = np.array(Image.open(os.path.join(data_dir, "masks", id_)).convert('L').resize((config.IM_WIDTH, config.IM_HEIGHT))).reshape((config.IM_HEIGHT, config.IM_WIDTH, 1)) / 255.0
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    return torch.tensor(X_train).permute(0, 3, 1, 2).to(config.DEVICE), torch.tensor(X_valid).permute(0, 3, 1, 2).to(config.DEVICE), torch.tensor(y_train).permute(0, 3, 1, 2).to(config.DEVICE), torch.tensor(y_valid).permute(0, 3, 1, 2).to(config.DEVICE)
