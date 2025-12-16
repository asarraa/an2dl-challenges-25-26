#!/usr/bin/env python3

import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# =============================================================================
# --- 1. DATASET CLASS MULTI-SCALA (CORRETTA) ---
# =============================================================================

class MultiScaleTileDataset(Dataset):
    """
    Dataset multi-scala che carica una coppia di tile (contesto e dettaglio).
    Restituisce 3 elementi: (img_contesto, img_dettaglio, label/nome_file).
    """
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform_context=None, transform_detail=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.context_dir = base_dir / "images_context" # Cartella per i tile 768px
        self.detail_dir = base_dir / "images_detail"   # Cartella per i tile 256px
        self.transform_context = transform_context
        self.transform_detail = transform_detail
        self.is_test = is_test

        # La normalizzazione è la stessa per entrambi i rami
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row['sample_index']
        
        context_path = self.context_dir / img_name
        detail_path = self.detail_dir / img_name
        
        try:
            img_context = Image.open(context_path).convert("RGB")
            img_detail = Image.open(detail_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not read image pair for {img_name}. Error: {e}")

        # Applica le augmentation
        if self.transform_context:
            img_context = self.transform_context(img_context)
        if self.transform_detail:
            img_detail = self.transform_detail(img_detail)

        # Trasforma in tensori e normalizza
        tensor_context = self.normalize(transforms.functional.to_tensor(img_context))
        tensor_detail = self.normalize(transforms.functional.to_tensor(img_detail))
        
        if self.is_test:
            return tensor_context, tensor_detail, img_name
        else:
            label = row['label']
            return tensor_context, tensor_detail, torch.tensor(label, dtype=torch.long)

# =============================================================================
# --- 2. FUNZIONI PER CREARE I DATALOADER ---
# =============================================================================

def get_multiscale_loaders(base_path: Path, batch_size: int, val_split: float = 0.2, seed: int = 42):
    """
    Crea e restituisce i DataLoader per training e validazione.
    """
    print("\n--- Creating Multi-Scale Train & Validation DataLoaders ---")
    train_val_dir = base_path / "train"
    csv_path = train_val_dir / "train_patches.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)

    # Mappatura automatica delle label da stringa a intero
    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    print(f"Labels mapped to integers: {label_map}")

    # Split stratificato basato sullo slide originale per evitare data leakage
    unique_slides_df = df.groupby('original_sample')['label'].first().reset_index()
    train_slides, val_slides = train_test_split(
        unique_slides_df, test_size=val_split, random_state=seed, stratify=unique_slides_df['label']
    )
    train_df = df[df['original_sample'].isin(train_slides['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides['original_sample'])]
    
    print(f"Data split result: {len(train_df)} train tiles, {len(val_df)} validation tiles.")

    # --- Augmentations ---
    # Definiamo augmentation separate per ogni scala
    train_augmentation_context = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    
    train_augmentation_detail = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])

    val_augmentation = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    
    # --- Create Datasets & DataLoaders ---
    train_ds = MultiScaleTileDataset(train_df, train_val_dir, transform_context=train_augmentation_context, transform_detail=train_augmentation_detail)
    val_ds = MultiScaleTileDataset(val_df, train_val_dir, transform_context=val_augmentation, transform_detail=val_augmentation)

    num_workers = min(os.cpu_count() or 2, 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_multiscale_test_loader(base_path: Path, batch_size: int):
    """
    Crea e restituisce il DataLoader per il test set.
    """
    print("\n--- Creating Multi-Scale Test DataLoader ---")
    test_dir = base_path / "test"
    csv_path = test_dir / "test_patches.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV file not found in '{csv_path}'.")
        
    df = pd.read_csv(csv_path)
    
    # Per il test, applichiamo solo il resize
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    
    test_ds = MultiScaleTileDataset(df, test_dir, transform_context=test_transform, transform_detail=test_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 2, 8), pin_memory=True)
    
    return test_loader

# =============================================================================
# --- 3. PIPELINE DI VERIFICA E VISUALIZZAZIONE ---
# =============================================================================

def verify_and_visualize(loader: DataLoader, num_images: int = 8, dataset_name: str = "Dataset"):
    """
    Estrae un batch, lo stampa e visualizza le immagini (sia contesto che dettaglio).
    """
    print(f"\n--- Verifying and Visualizing a batch from the {dataset_name} ---")
    
    # Estrai un batch. Ora sono 3 elementi.
    context_batch, detail_batch, labels_or_names = next(iter(loader))
    
    print(f"  Context Batch Shape: {context_batch.shape}")
    print(f"  Detail Batch Shape: {detail_batch.shape}")
    if isinstance(labels_or_names, torch.Tensor):
        print(f"  Labels in batch: {labels_or_names.numpy()}")

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    # Visualizza immagini di contesto e dettaglio una sotto l'altra
    fig, axs = plt.subplots(2, num_images, figsize=(20, 5))
    fig.suptitle(f"Sample Batch from {dataset_name}", fontsize=16)

    for i in range(min(num_images, len(context_batch))):
        # Immagine di Contesto
        img_ctx = context_batch[i].numpy().transpose((1, 2, 0)); img_ctx = std * img_ctx + mean; img_ctx = np.clip(img_ctx, 0, 1)
        axs[0, i].imshow(img_ctx)
        axs[0, i].set_title(f"Context\nLabel: {labels_or_names[i].item() if isinstance(labels_or_names, torch.Tensor) else 'N/A'}")
        axs[0, i].axis("off")
        
        # Immagine di Dettaglio
        img_det = detail_batch[i].numpy().transpose((1, 2, 0)); img_det = std * img_det + mean; img_det = np.clip(img_det, 0, 1)
        axs[1, i].imshow(img_det)
        axs[1, i].set_title(f"Detail")
        axs[1, i].axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# --- 4. ENTRY POINT (PER LA VERIFICA DIRETTA) ---
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verification pipeline for multi-scale data loaders.")
    parser.add_argument('--base_path', type=Path, required=True, help='Path to the multi-scale preprocessed data directory.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for the DataLoaders.')
    args = parser.parse_args()
    
    try:
        train_loader, val_loader = get_multiscale_loaders(base_path=args.base_path, batch_size=args.batch_size)
        verify_and_visualize(train_loader, dataset_name="Training Set")
        verify_and_visualize(val_loader, dataset_name="Validation Set")
        
        test_loader = get_multiscale_test_loader(base_path=args.base_path, batch_size=args.batch_size)
        verify_and_visualize(test_loader, dataset_name="Test Set")
        
    except Exception as e:
        import traceback; traceback.print_exc()