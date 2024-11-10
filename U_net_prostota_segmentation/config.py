# config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dados"
IM_HEIGHT = 128
IM_WIDTH = 128
N_FILTERS = 16
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MODEL_SAVE_PATH = "unet_trained_model.pth"
