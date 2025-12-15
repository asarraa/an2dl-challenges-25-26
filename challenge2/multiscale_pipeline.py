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

# =============================================================================
# --- 1. DATASET CLASS MULTI-SCALA ---
# =============================================================================
class MultiScaleTileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_images_dir: Path, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        # Le cartelle delle immagini ora hanno un suffisso che definisce la scala
        self.context_dir = base_images_dir.parent / "images_context" # 768px
        self.detail_dir = base_images_dir.parent / "images_detail"   # 256px
        self.transform = transform
        self.is_test = is_test

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row['sample_index']
        
        # Carica entrambe le immagini
        context_path = self.context_dir / img_name
        detail_path = self.detail_dir / img_name
        
        try:
            img_context = Image.open(context_path).convert("RGB")
            img_detail = Image.open(detail_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not read image pair for {img_name}. Error: {e}")

        # Applica le augmentation (devono essere applicate a entrambe le immagini in modo identico!)
        if self.transform:
            # Per trasformazioni geometriche identiche, le passiamo come un dizionario
            sample = {'image_context': img_context, 'image_detail': img_detail}
            transformed_sample = self.transform(sample)
            img_context = transformed_sample['image_context']
            img_detail = transformed_sample['image_detail']

        # Trasforma in tensori e normalizza
        tensor_context = self.normalize(transforms.functional.to_tensor(img_context))
        tensor_detail = self.normalize(transforms.functional.to_tensor(img_detail))
        
        if self.is_test:
            return (tensor_context, tensor_detail), img_name
        else:
            label = row['label']
            return (tensor_context, tensor_detail), torch.tensor(label, dtype=torch.long)

# =============================================================================
# --- 2. LOGICA PER I LOADER ---
# =============================================================================
def get_multiscale_loaders(base_path: Path, batch_size: int, val_split: float = 0.2, seed: int = 42):
    print("\n--- Creating Multi-Scale DataLoaders ---")
    train_val_dir = base_path / "train"
    # Diamo un percorso di base, il Dataset troverà le cartelle _context e _detail
    images_dir = train_val_dir / "images_base" 
    csv_path = train_val_dir / "train_patches.csv"

    df = pd.read_csv(csv_path)
    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    print(f"Labels mapped: {label_map}")

    # (La logica di split rimane la stessa)
    unique_slides_df = df.groupby('original_sample')['label'].first().reset_index()
    train_slides, val_slides = train_test_split(unique_slides_df, test_size=val_split, random_state=seed, stratify=unique_slides_df['label'])
    train_df = df[df['original_sample'].isin(train_slides['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides['original_sample'])]
    
    # --- Augmentations personalizzate per multi-scala ---
    # Creiamo una classe per assicurarci che le augmentation geometriche siano identiche
    class ApplyToBoth:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, sample):
            return {key: self.transform(img) for key, img in sample.items()}

    train_augmentation = transforms.Compose([
        ApplyToBoth(transforms.RandomHorizontalFlip(p=0.5)),
        ApplyToBoth(transforms.RandomVerticalFlip(p=0.5)),
        # La rotazione e il crop sono i più importanti da sincronizzare
        ApplyToBoth(transforms.RandomRotation(degrees=30)),
        # Il ColorJitter può essere applicato separatamente
        transforms.Lambda(lambda sample: {
            'image_context': transforms.ColorJitter(brightness=0.1, contrast=0.1)(sample['image_context']),
            'image_detail': transforms.ColorJitter(brightness=0.1, contrast=0.1)(sample['image_detail'])
        }),
        # Resize finale
        transforms.Lambda(lambda sample: {
            'image_context': transforms.Resize((224, 224))(sample['image_context']),
            'image_detail': transforms.Resize((224, 224))(sample['image_detail'])
        }),
    ])
    
    val_augmentation = transforms.Compose([
        transforms.Lambda(lambda sample: {
            'image_context': transforms.Resize((224, 224))(sample['image_context']),
            'image_detail': transforms.Resize((224, 224))(sample['image_detail'])
        }),
    ])
    
    # --- Create Datasets & DataLoaders ---
    train_ds = MultiScaleTileDataset(train_df, images_dir, transform=train_augmentation)
    val_ds = MultiScaleTileDataset(val_df, images_dir, transform=val_augmentation)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader