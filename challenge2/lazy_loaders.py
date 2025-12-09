# Importing OpenCV for image loading and processing
import cv2
# OS utilities for system info and file manipulation
import os
# Numpy for array operations
import numpy as np
# PyTorch main library for tensors and neural network utilities
import torch
# Pandas for reading and handling CSV metadata
import pandas as pd
# Pillow for converting arrays to PIL Image objects
from PIL import Image
# Pathlib for cleaner filesystem path manipulation
from pathlib import Path
# Torchvision v2 transforms (optimized image transforms)
from torchvision.transforms import v2 as transforms
# Scikit-learn utility for splitting dataset into train/val
from sklearn.model_selection import train_test_split
# PyTorch DataLoader for batching and multiprocessing loading
from torch.utils.data import DataLoader
# External configuration file containing LOADER_PARAMS dict
from config import LOADER_PARAMS

# Global random seed for reproducibility
SEED = 42


class LazyImageDataset(torch.utils.data.Dataset):
    """
    Memory-efficient Dataset that loads images from disk on-demand.
    
    Args:
        csv_df (pd.DataFrame): DataFrame with 'sample_index' and 'label' columns
        images_dir (Path): Directory containing image files
        masks_dir (Path, optional): Directory containing mask files (same names as images).
        add_mask_channel (bool): If True, appends mask as 4th channel to the image.
        transform (callable, optional): Transforms to apply
    """
    def __init__(self, csv_df, images_dir, masks_dir=None, add_mask_channel=False, transform=None):
        # Copy of dataset annotation CSV; reset index for proper indexing
        self.csv_df = csv_df.reset_index(drop=True)
        # Directory containing images
        self.images_dir = Path(images_dir)
        # Optional directory containing masks
        self.masks_dir = Path(masks_dir) if masks_dir else None
        # Whether to append mask as 4th image channel
        self.add_mask_channel = add_mask_channel
        # Optional augmentations or preprocessing transforms
        self.transform = transform
        
        # Base transform always used (convert image to tensor)
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),                                 # Convert PIL/numpy to Torch tensor
            transforms.ToDtype(torch.float32, scale=True)         # Convert dtype and scale to [0,1]
        ])
    
    def __len__(self):
        # Total number of samples (rows in CSV)
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        # Load a specific sample's metadata from the CSV
        row = self.csv_df.iloc[idx]
        # Column 'sample_index' contains filename
        img_name = row['sample_index']
        # Column 'label' contains class index
        label = row['label']
        
        # Build full path to image on disk and read with OpenCV
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        
        # If OpenCV failed to load the image, raise error
        if image is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        
        # Convert image format from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If mask channel is enabled, load and concatenate the mask
        if self.add_mask_channel:
            # Check masks_dir exists when needed
            if self.masks_dir is None:
                raise ValueError("masks_dir must be provided when add_mask_channel=True")
            # Build full path to mask and load
            mask_path = self.masks_dir / img_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale
            
            # Error if mask file is missing
            if mask is None:
                raise FileNotFoundError(f"Could not load mask {mask_path}")
            # Resize mask if spatial dimensions differ from image
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Stack mask as 4th channel (RGB + Mask → 4 channels)
            image = np.dstack((image, mask))
        
        # Convert numpy array to PIL Image (required for torchvision v2 transforms)
        image_pil = Image.fromarray(image)
        
        # Convert PIL image into normalized torch tensor
        image_tensor = self.to_tensor(image_pil)
        
        # Apply augmentations if provided (only applied to training)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Return the image tensor and label as torch.long type
        return image_tensor, torch.tensor(label, dtype=torch.long)


def make_loader(ds, batch_size, shuffle, drop_last):
    """
    Create a PyTorch DataLoader with optimized settings.
    """
    # Detect number of CPU cores available
    cpu_cores = os.cpu_count() or 2
    # Choose number of workers between 2 and 8 based on CPU count
    num_workers = max(2, min(8, cpu_cores))  # Increased workers for disk I/O
    
    # Create DataLoader optimized for GPU training and disk I/O heavy datasets
    return DataLoader(
        ds,                         # Dataset object
        batch_size=batch_size,      # Number of samples per batch
        shuffle=shuffle,            # Shuffle (True for training)
        drop_last=drop_last,        # Drop final batch if smaller size
        num_workers=num_workers,    # Parallel workers for loading data
        pin_memory=True,            # Speeds transfer to GPU
        pin_memory_device="cuda" if torch.cuda.is_available() else "",  # Specify device for pinning
        prefetch_factor=4,          # Preload samples into worker queue
        persistent_workers=True,    # Keep workers alive between epochs
    )


def get_loaders(augmentation=None, batch_size=LOADER_PARAMS["batch_size"], base_path=None, add_mask_channel=False):
    """
    Create train and validation dataloaders using lazy loading (no RAM overload).
    
    Returns:
        train_loader, val_loader, input_shape
    """
    # Base dataset path; fallback to default if none provided
    base_path = Path(base_path) if base_path else Path("../../drive/MyDrive/AN2DL_Challenge2-TheBigBatchTheory/data/dataset/testpreprocessing")
    # Subdirectories containing training images and (optionally) masks
    images_dir = base_path / "train" / "images"
    masks_dir = base_path / "train" / "masks" if add_mask_channel else None
    # Path to CSV that maps image filenames to labels
    csv_path = base_path / "train_patches.csv"
    
    # Read CSV metadata into memory (very small footprint)
    df = pd.read_csv(csv_path)
    
    # Create mapping: string labels → integer labels
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    # Replace labels in dataframe with integer mapping
    df['label'] = df['label'].map(label_map)
    
    # Split dataset into train and validation sets (stratified by label distribution)
    train_df, val_df = train_test_split(
        df,
        test_size=LOADER_PARAMS["percentage_validation"],  # percentage for validation in config
        random_state=SEED,                                  # reproducible split
        stratify=df['label']                                # maintain class balance
    )
    
    # Load one sample image for determining input shape
    sample_path = images_dir / df.iloc[0]['sample_index']
    sample_img = cv2.imread(str(sample_path))
    if sample_img is None:
        raise FileNotFoundError(f"Could not load sample image {sample_path}")
    
    # Determine shape depending on whether masks are appended
    if add_mask_channel:
        sample_mask = cv2.imread(str(masks_dir / df.iloc[0]['sample_index']), cv2.IMREAD_GRAYSCALE)
        if sample_mask is None:
            raise FileNotFoundError(f"Could not load sample mask {masks_dir / df.iloc[0]['sample_index']}")
        # Input shape becomes 4 channels (RGB + Mask)
        input_shape = (4, sample_img.shape[0], sample_img.shape[1])  # (C, H, W)
    else:
        # Standard shape: channels, height, width
        input_shape = (sample_img.shape[2], sample_img.shape[0], sample_img.shape[1])  # (C, H, W)
    
    # Define default training augmentations if none passed by user
    if augmentation is None:
        train_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),                                   # 50% chance of horizontal flip
            transforms.RandomVerticalFlip(p=0.5),                                     # 50% chance of vertical flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),     # Random color perturbations
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)) # Random rotations/scaling
        ])
    else:
        # Use custom augmentations if provided
        train_augmentation = augmentation
    
    # Create training dataset with augmentations
    train_ds = LazyImageDataset(train_df, images_dir, masks_dir=masks_dir, add_mask_channel=add_mask_channel, transform=train_augmentation)
    # Validation dataset without augmentations
    val_ds = LazyImageDataset(val_df, images_dir, masks_dir=masks_dir, add_mask_channel=add_mask_channel, transform=None)
    
    # Create DataLoaders for training and validation
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Print dataset statistics
    print(f"✅ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"✅ Input shape: {input_shape}")
    
    # Return both dataloaders and input image shape
    return train_loader, val_loader, input_shape
