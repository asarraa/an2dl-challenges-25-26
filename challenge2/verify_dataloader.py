#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import argparse
import os

# =============================================================================
# 1. THE ROBUST DATASET CLASS
# =============================================================================

class HistologyTileDataset(Dataset):
    """
    Robust, lazy-loading Dataset for histology tiles.
    Handles training/validation (with labels and transforms) and testing (without).
    """
    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.is_test = is_test

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row['sample_index']
        img_path = self.images_dir / img_name
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not read image {img_path}. Error: {e}")

        if self.transform:
            image = self.transform(image)
        
        image_tensor = transforms.functional.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        
        if self.is_test:
            return image_tensor, img_name
        else:
            label = row['label']
            return image_tensor, torch.tensor(label, dtype=torch.long)

# =============================================================================
# 2. DATA SPLITTING & LOADER CREATION (THE FUNCTIONS YOU NEED)
# =============================================================================

def verify_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Checks for data leakage between the training and validation sets."""
    print("\n--- Verifying Data Split Integrity (Checking for Leakage) ---")
    train_slides = set(train_df['original_sample'].unique())
    val_slides = set(val_df['original_sample'].unique())
    leakage = train_slides.intersection(val_slides)
    
    if not leakage:
        print("✅ PASSED: No data leakage detected.")
    else:
        print(f"❌ FAILED: Data leakage detected! {len(leakage)} slides are in both sets.")

def get_loaders(
    base_path: Path,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    # Kept for compatibility, but recommend not using it.
    # The new Dataset assumes 3 channels (RGB).
    add_mask_channel: bool = False
):
    """
    Creates and returns train and validation DataLoaders, input_shape, and class_weights.
    """
    print("\n--- Creating Training & Validation DataLoaders ---")
    if add_mask_channel:
        print("⚠️ WARNING: add_mask_channel is not fully supported in this version and assumes 3-channel RGB input.")

    train_val_dir = base_path / "train"
    images_dir = train_val_dir / "images"
    csv_path = train_val_dir / "train_patches.csv"

    if not all([images_dir.exists(), csv_path.exists()]):
        raise FileNotFoundError(f"Required files not found. Ensure '{images_dir}' and '{csv_path}' exist.")

    df = pd.read_csv(csv_path)

    # --- Map String Labels to Integers ---
    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    print(f"Labels mapped to integers: {label_map}")

    # --- Stratified split based on ORIGINAL slide ---
    unique_slides_df = df.groupby('original_sample')['label'].first().reset_index()
    train_slides, val_slides = train_test_split(
        unique_slides_df, # Split the whole dataframe to keep labels for stratification
        test_size=val_split,
        random_state=seed,
        stratify=unique_slides_df['label']
    )
    
    train_df = df[df['original_sample'].isin(train_slides['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides['original_sample'])]
    verify_split_integrity(train_df, val_df)
    
    print(f"Data split result: {len(train_df)} train tiles, {len(val_df)} validation tiles.")

    # --- Class Weights Calculation (on training tiles) ---
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label'].to_numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Calculated class weights for training set: {class_weights.numpy()}")

    # --- Augmentations ---
    train_augmentation = transforms.Compose([
        transforms.Resize((224, 224)),
       # transforms.RandomHorizontalFlip(p=0.5),
       # transforms.RandomVerticalFlip(p=0.5),
       # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    val_augmentation = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    # --- Create Datasets & DataLoaders ---
    train_ds = HistologyTileDataset(train_df, images_dir, transform=train_augmentation)
    val_ds = HistologyTileDataset(val_df, images_dir, transform=val_augmentation)

    num_workers = os.cpu_count() or 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # --- Determine Input Shape ---
    # We know it will be 3 channels, 224x224 after transforms
    input_shape = (3, 224, 224)
    
    return train_loader, val_loader, input_shape, class_weights

def get_test_loader(
    base_path: Path,
    batch_size: int,
    add_mask_channel: bool = False
):
    """Creates and returns the DataLoader for the test set."""
    print("\n--- Creating Test DataLoader ---")
    if add_mask_channel:
        print("⚠️ WARNING: add_mask_channel is not fully supported and assumes 3-channel RGB input.")
        
    test_dir = base_path / "test"
    images_dir = test_dir / "images"
    csv_path = test_dir / "test_patches.csv"
    
    if not all([images_dir.exists(), csv_path.exists()]):
        raise FileNotFoundError(f"Required files for test set not found in '{test_dir}'.")
        
    df = pd.read_csv(csv_path)
    
    test_ds = HistologyTileDataset(df, images_dir, transform=transforms.Resize((224, 224)), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 2, pin_memory=True)
    
    input_shape = (3, 224, 224)
    
    return test_loader, input_shape

# =============================================================================
# 3. VERIFICATION & VISUALIZATION PIPELINE
# =============================================================================

def verify_and_visualize(loader: DataLoader, num_images: int = 8, dataset_name: str = "Dataset", output_dir: Path = Path('.')):
    """Pulls one batch and saves a plot of the images and labels."""
    # This function remains the same as before
    print(f"\n--- Verifying and Visualizing a batch from the {dataset_name} ---")
    images, labels = next(iter(loader))
    print(f"  Batch shape: {images.shape}, Labels: {labels.numpy()}")
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    plt.figure(figsize=(16, 8))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(2, max(1, num_images // 2), i + 1)
        img = images[i].numpy().transpose((1, 2, 0)); img = std * img + mean; img = np.clip(img, 0, 1)
        plt.imshow(img); ax.set_title(f"Label: {labels[i].item()}"); ax.axis("off")
    plt.suptitle(f"Sample Batch from {dataset_name}", fontsize=16)
    output_filename = output_dir / f"verification_batch_{dataset_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_filename); print(f"✅ Saved verification plot to: {output_filename}"); plt.close()

# =============================================================================
# 4. SCRIPT ENTRY POINT (FOR DIRECT VERIFICATION)
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verification pipeline for histology data loaders.")
    parser.add_argument('--base_path', type=Path, required=True, help='Path to the preprocessed data directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the DataLoaders.')
    args = parser.parse_args()
    
    try:
        # --- Run the full loading and verification process ---
        train_loader, val_loader, _, _ = get_loaders(base_path=args.base_path, batch_size=args.batch_size)
        verify_and_visualize(train_loader, dataset_name="Training Set", output_dir=args.base_path)
        verify_and_visualize(val_loader, dataset_name="Validation Set", output_dir=args.base_path)
        
        test_loader, _ = get_test_loader(base_path=args.base_path, batch_size=args.batch_size)
        print("\n--- Verifying a batch from the Test Set ---")
        images, filenames = next(iter(test_loader))
        print(f"  Test batch shape: {images.shape}, Filenames of first 4: {filenames[:4]}")
        print("\n✅ All loaders created and verified successfully.")
        
    except Exception as e:
        import traceback; traceback.print_exc()