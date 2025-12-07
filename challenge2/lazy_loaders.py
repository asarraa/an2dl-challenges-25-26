import cv2
import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2 as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import LOADER_PARAMS

SEED = 42

class LazyImageDataset(torch.utils.data.Dataset):
    """
    Memory-efficient Dataset that loads images from disk on-demand.
    
    Args:
        csv_df (pd.DataFrame): DataFrame with 'sample_index' and 'label' columns
        images_dir (Path): Directory containing image files
        transform (callable, optional): Transforms to apply
    """
    def __init__(self, csv_df, images_dir, transform=None):
        self.csv_df = csv_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Base transform: convert to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
    
    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        # Get metadata
        row = self.csv_df.iloc[idx]
        img_name = row['sample_index']
        label = row['label']
        
        # Load image from disk (lazy loading)
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms v2
        image_pil = Image.fromarray(image)
        
        # Convert to tensor
        image_tensor = self.to_tensor(image_pil)
        
        # Apply augmentations
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


def make_loader(ds, batch_size, shuffle, drop_last):
    """
    Create a PyTorch DataLoader with optimized settings.
    """
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_cores))  # Increased workers for disk I/O
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4,
        persistent_workers=True,  # Keep workers alive between epochs
    )


def get_loaders(augmentation=None, batch_size=LOADER_PARAMS["batch_size"]):
    """
    Create train and validation dataloaders using lazy loading (no RAM overload).
    
    Returns:
        train_loader, val_loader, input_shape
    """
    # Paths
    base_path = Path("../../drive/MyDrive/AN2DL_Challenge2-TheBigBatchTheory/data/dataset/testpreprocessing")
    images_dir = base_path / "train" / "images"
    csv_path = base_path / "train_patches.csv"
    
    # Load CSV metadata (small, fits in RAM)
    df = pd.read_csv(csv_path)
    
    # Map string labels to integers
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_map)
    
    # Split into train/val
    train_df, val_df = train_test_split(
        df,
        test_size=LOADER_PARAMS["percentage_validation"],
        random_state=SEED,
        stratify=df['label']
    )
    
    # Determine input shape from one sample
    sample_img = cv2.imread(str(images_dir / df.iloc[0]['sample_index']))
    input_shape = (sample_img.shape[2], sample_img.shape[0], sample_img.shape[1])  # (C, H, W)
    
    # Define augmentations
    if augmentation is None:
        train_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
    else:
        train_augmentation = augmentation
    
    # Create lazy datasets
    train_ds = LazyImageDataset(train_df, images_dir, transform=train_augmentation)
    val_ds = LazyImageDataset(val_df, images_dir, transform=None)
    
    # Create loaders
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"✅ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"✅ Input shape: {input_shape}")
    
    return train_loader, val_loader, input_shape