import cv2
import os
import numpy as np
import torch
import pandas as pd
from torchvision.transforms import v2 as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import LOADER_PARAMS
SEED = 42

# Custom Dataset class that applies transforms v2 on-the-fly
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that applies data augmentation transforms using transforms v2.

    Following the recommended approach from torchvision documentation:
    - Use ToImage() to convert PIL to tensor
    - Use ToDtype(torch.float32, scale=True) to convert to float and scale to [0, 1]

    Args:
        data (np.ndarray): Input images with shape (N, H, W, C)
        labels (np.ndarray): Labels with shape (N,)
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, data, labels, transform=None): #transform here is your "compose"
        self.data = data
        self.labels = labels
        self.transform = transform

        # Base transform: convert to tensor (following v2 guidelines)
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label
        image = self.data[idx]
        label = self.labels[idx]

        # Convert numpy to PIL Image
        image_pil = Image.fromarray((image * 255).astype(np.uint8))

        # Convert to tensor using v2 recommended approach
        image_tensor = self.to_tensor(image_pil)

        # Apply additional transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)

def make_loader(ds, batch_size, shuffle, drop_last):
    """
    Create a PyTorch DataLoader with optimized settings.

    Args:
        ds (Dataset): PyTorch Dataset object
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle data at each epoch
        drop_last (bool): Whether to drop last incomplete batch

    Returns:
        DataLoader: Configured DataLoader instance
    """
    # Determine optimal number of worker processes for data loading
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(4, cpu_cores))

    # Create DataLoader with performance optimizations
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4,  # Load 4 batches ahead
    )

def get_loaders(augmentation = None, batch_size=LOADER_PARAMS["batch_size"]):
    path = "./data/testpreprocessing"
    X = np.load(os.path.join(path, "/processed_patches.npy"))
    y = pd.read_csv(os.path.join(path, "/train_patches.csv"))
    
    # Load dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=SEED, test_size=np.floor(LOADER_PARAMS["percentage_validation"]*len(X)), stratify=y
    )
    
    # Define the input shape based on the training data
    input_shape = (X_train.shape[3], X_train.shape[1], X_train.shape[2])
    
    # Define data augmentation pipeline for training using transforms v2
    if augmentation is None:
        train_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.75, saturation=0.75),
        transforms.RandomAffine(degrees=72, translate=(0.2, 0.2), scale=(0.8, 1.2))
        #transforms.RandomErasing(p=1.0, scale=(0.01, 0.033), value='random')    for _ in range(5)
        ])
    
    # Create augmented datasets
    train_aug_ds = AugmentedDataset(X_train, y_train.squeeze(), transform=train_augmentation)
    val_aug_ds = AugmentedDataset(X_val, y_val.squeeze(), transform=None)
    #test_aug_ds = AugmentedDataset(X_test, y_test.squeeze(), transform=None)
    
    # Create data loaders for augmented datasets
    train_aug_loader = make_loader(train_aug_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_aug_loader = make_loader(val_aug_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    #test_aug_loader = make_loader(test_aug_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_aug_loader, val_aug_loader, input_shape