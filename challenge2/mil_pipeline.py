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
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# --- 1. DATASET CLASS PER MULTIPLE INSTANCE LEARNING (MIL) ---
# =============================================================================

class MILDataset(Dataset):
    """
    Dataset per Multiple Instance Learning.
    Ogni __getitem__ restituisce un intero "sacco" (bag) di tile per una slide.
    """
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None, is_test: bool = False):
        self.base_dir = base_dir
        self.transform = transform
        self.is_test = is_test

        self.slides_df = df.groupby('original_sample').apply(lambda x: x['sample_index'].tolist()).reset_index(name='tiles')
        if not is_test:
            labels = df.groupby('original_sample')['label'].first()
            self.slides_df = self.slides_df.merge(labels, on='original_sample')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.slides_df)
    
    def __getitem__(self, idx: int):
        slide_info = self.slides_df.iloc[idx]
        slide_name = slide_info['original_sample']
        tile_names = slide_info['tiles']
        
        # Puntiamo direttamente alla cartella 'images_detail'
        img_dir = self.base_dir / "images_detail"
        
        bag_of_tiles = []
        for tile_name in tile_names:
            tile_path = img_dir / tile_name
            try:
                img = Image.open(tile_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                tensor = self.normalize(transforms.functional.to_tensor(img))
                bag_of_tiles.append(tensor)
            except Exception:
                continue
        
        if not bag_of_tiles:
            return torch.empty(0, 3, 224, 224), (slide_name if self.is_test else torch.tensor(-1, dtype=torch.long))

        bag_tensor = torch.stack(bag_of_tiles, dim=0)

        if self.is_test:
            return bag_tensor, slide_name
        else:
            label = slide_info['label']
            return bag_tensor, torch.tensor(label, dtype=torch.long)

def mil_collate_fn(batch):
    bags = [item[0] for item in batch if item[0].size(0) > 0] # Filtra sacchi vuoti
    labels_or_names = [item[1] for item in batch if item[0].size(0) > 0]
    
    if not bags: # Se l'intero batch è vuoto
        return [], []

    if isinstance(labels_or_names[0], torch.Tensor):
        labels_or_names = torch.stack(labels_or_names, dim=0)
        
    return bags, labels_or_names

# =============================================================================
# --- 2. FUNZIONI PER CREARE I DATALOADER (CON TEST LOADER INCLUSO) ---
# =============================================================================
def get_mil_loaders(base_path: Path, batch_size: int, val_split: float = 0.2, seed: int = 42, augmentation=None):
    train_val_dir = base_path / "train"
    csv_path = train_val_dir / "train_patches.csv"
    df = pd.read_csv(csv_path)

    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    unique_slides_df = df[['original_sample', 'label']].drop_duplicates()
    train_slides_df, val_slides_df = train_test_split(unique_slides_df, test_size=val_split, random_state=seed, stratify=unique_slides_df['label'])

    train_df = df[df['original_sample'].isin(train_slides_df['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides_df['original_sample'])]
    
    if augmentation is not None:
        train_augmentation = augmentation
    else:
        train_augmentation = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1)])
    val_augmentation = transforms.Compose([transforms.Resize((224, 224))])

    train_ds = MILDataset(train_df, train_val_dir, transform=train_augmentation)
    val_ds = MILDataset(val_df, train_val_dir, transform=val_augmentation)

    num_workers = min(os.cpu_count() or 2, 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=mil_collate_fn)
    
    return train_loader, val_loader

# --- NUOVA FUNZIONE AGGIUNTA ---
def get_mil_test_loader(base_path: Path, batch_size: int):
    """
    Crea e restituisce il DataLoader per il test set per il modello MIL.
    """
    print("\n--- Creating MIL Test DataLoader ---")
    test_dir = base_path / "test"
    csv_path = test_dir / "test_patches.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV file not found in '{csv_path}'.")
        
    df = pd.read_csv(csv_path)
    
    # Per il test, applichiamo solo il resize
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    
    test_ds = MILDataset(df, test_dir, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 2, 4), collate_fn=mil_collate_fn)
    
    return test_loader
# --- FINE NUOVA FUNZIONE ---


# =============================================================================
# --- 3. PIPELINE DI VERIFICA E VISUALIZZAZIONE (Invariata) ---
# =============================================================================
# ... (la funzione verify_and_visualize rimane la stessa, ma ora non la useremo)

# =============================================================================
# --- 4. ENTRY POINT (Invariato) ---
# =============================================================================
if __name__ == '__main__':
    # ... (la logica di argparse rimane la stessa)
    pass