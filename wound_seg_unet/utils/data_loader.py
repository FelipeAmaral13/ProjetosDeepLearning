import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class WoundDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.img_size = img_size
        
        # Transformações para imagens
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transformações para máscaras
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Carregar imagem e máscara
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Aplicar transformações
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask

def get_dataloaders(data_dir, batch_size=8, img_size=256):
    train_dataset = WoundDataset(
        img_dir=os.path.join(data_dir, 'train', 'images'),
        mask_dir=os.path.join(data_dir, 'train', 'labels'),
        img_size=img_size
    )
    
    val_dataset = WoundDataset(
        img_dir=os.path.join(data_dir, 'validation', 'images'),
        mask_dir=os.path.join(data_dir, 'validation', 'labels'),
        img_size=img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader