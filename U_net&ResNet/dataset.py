import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from typing import List, Tuple

class CustomDataset(TorchDataset):
    """
    Custom dataset for loading image and mask pairs from directories.

    Args:
        image_dir (str): Directory containing images.
        mask_dir (str): Directory containing masks.
        dataset_type (str): 'train' or 'val/test' to specify the split.
        split_ratio (float): Split ratio for train/val/test.
        transform (callable, optional): Transform to be applied to both image and mask.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        dataset_type: str = "train",
        split_ratio: float = 0.2,
        transform: callable = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = os.listdir(mask_dir)

        if dataset_type == "train":
            split_index = int(len(self.masks) * (1 - split_ratio))
            self.masks = self.masks[:split_index]
        else:
            split_index = int(len(self.masks) * (1 - split_ratio))
            self.masks = self.masks[split_index:]

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        img_path = os.path.join(self.image_dir, self.masks[index].replace("_mask.png", ".png"))

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        except Exception as e:
            raise Exception(f"Error loading image/mask at index {index}: {str(e)}")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
    def plot_sample(self, index: int):
        """
        Plot a sample image and its corresponding mask.

        Args:
            index (int): Index of the sample to plot.
        """
        image, mask = self.__getitem__(index)

        # Converte o tensor de imagem para um array NumPy e ajusta o formato
        image_numpy = image.permute(1, 2, 0).cpu().numpy()

        # Converte o tensor de máscara para um array NumPy e ajusta o formato
        mask_numpy = mask.cpu().numpy()

        # Converte a máscara para uma imagem binária
        mask_image = Image.fromarray((mask_numpy * 255).astype(np.uint8))

        # Plotagem da imagem original
        plt.subplot(1, 2, 1)
        plt.imshow(image_numpy)
        plt.title("Image")

        # Plotagem da máscara
        plt.subplot(1, 2, 2)
        plt.imshow(mask_image, cmap="gray")
        plt.title("Mask")

        plt.tight_layout()
        plt.show()


