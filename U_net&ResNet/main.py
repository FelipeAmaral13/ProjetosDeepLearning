from torch.utils.data import DataLoader
from dataset import CustomDataset
from u_net import UNet
from resnet_unet import ResNeXtUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt



def get_loaders(image_dir, mask_dir, batch_size, train_transform, val_transform, num_workers=2, pin_memory=True):

    train_ds = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        dataset_type="train",
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        dataset_type="test",
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

img_h, img_w = 224, 224

train_transform = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=img_h, width=img_w),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

img_dir = 'CXR_png'
mask_dir = 'masks'
batch_size = 4
pin_mem = True
workers = 2

train_loader, val_loader = get_loaders(img_dir, mask_dir, batch_size, train_transform, val_transforms, workers, pin_mem)

model = ResNeXtUNet(n_classes=1)
model.train_model(num_epochs=10, train_loader=train_loader, learning_rate=0.001, weight_decay=1e-3)
predicted_mask = model.predict_image(r"D:\Estudos\DSA\env_pytorch\U-net\CHNCXR_0001_0.png")
plt.imshow(predicted_mask[0, 0].cpu().numpy(), cmap="gray")
plt.title("Predicted Mask")
plt.show()