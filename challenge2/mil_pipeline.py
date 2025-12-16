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

        # Raggruppiamo il dataframe per slide. Ogni riga ora rappresenta una slide.
        self.slides_df = df.groupby('original_sample').apply(lambda x: x['sample_index'].tolist()).reset_index(name='tiles')
        if not is_test:
            labels = df.groupby('original_sample')['label'].first()
            self.slides_df = self.slides_df.merge(labels, on='original_sample')

    def __len__(self):
        return len(self.slides_df)
    
    def __getitem__(self, idx: int):
        slide_info = self.slides_df.iloc[idx]
        slide_name = slide_info['original_sample']
        tile_names = slide_info['tiles']
        
        # Per il MIL, usiamo solo una scala per semplicità. Iniziamo con i tile 'detail' (256px).
        img_dir = self.base_dir / "images_detail"
        
        bag_of_tiles = []
        for tile_name in tile_names:
            tile_path = img_dir / tile_name
            try:
                # Carica l'immagine in formato PIL, come richiesto da torchvision.transforms
                img = Image.open(tile_path).convert("RGB")
                
                # Applica le trasformazioni (che includono resize, ToTensor e normalize)
                if self.transform:
                    tensor = self.transform(img)
                else:
                    # Fallback nel caso non ci siano trasformazioni
                    tensor = transforms.functional.to_tensor(img)

                bag_of_tiles.append(tensor)
            except Exception as e:
                # Stampa un avviso se un tile non può essere caricato, ma non blocca l'esecuzione
                # print(f"Warning: Could not load tile {tile_path}, skipping. Error: {e}")
                continue
        
        # Se nessun tile per una slide è stato caricato, restituisci un tensore vuoto
        if not bag_of_tiles:
            return torch.empty(0, 3, 224, 224), (slide_name if self.is_test else torch.tensor(-1, dtype=torch.long))

        # "Impila" tutti i tensori dei tile in un unico grande tensore per la slide
        bag_tensor = torch.stack(bag_of_tiles, dim=0)

        if self.is_test:
            return bag_tensor, slide_name
        else:
            label = slide_info['label']
            return bag_tensor, torch.tensor(label, dtype=torch.long)

# Funzione 'collate' personalizzata per gestire sacchi (bag) di dimensioni diverse
def mil_collate_fn(batch):
    """
    Funzione per assemblare i batch nel DataLoader MIL.
    Gestisce il fatto che ogni campione (slide) ha un numero diverso di tile.
    """
    # Filtra eventuali campioni che hanno restituito sacchi vuoti
    batch = [item for item in batch if item[0].size(0) > 0]
    if not batch:
        return [], []

    bags = [item[0] for item in batch]
    labels_or_names = [item[1] for item in batch]
    
    # Le label vengono impilate in un unico tensore, i 'bags' rimangono una lista
    if isinstance(labels_or_names[0], torch.Tensor):
        labels_or_names = torch.stack(labels_or_names, dim=0)
        
    return bags, labels_or_names

# =============================================================================
# --- 2. FUNZIONI PER CREARE I DATALOADER (AGGIORNATE PER MIL) ---
# =============================================================================
def get_mil_loaders(base_path: Path, batch_size: int, uni_transform, val_split: float = 0.2, seed: int = 42):
    """
    Crea e restituisce i DataLoader per training e validazione per il modello MIL,
    usando le trasformazioni specifiche del modello (es. UNI).
    """
    train_val_dir = base_path / "train"
    csv_path = train_val_dir / "train_patches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at '{csv_path}'.")
    df = pd.read_csv(csv_path)

    # Mappatura automatica delle label
    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    
    # Split stratificato a livello di slide
    unique_slides_df = df[['original_sample', 'label']].drop_duplicates()
    train_slides_df, val_slides_df = train_test_split(unique_slides_df, test_size=val_split, random_state=seed, stratify=unique_slides_df['label'])
    train_df = df[df['original_sample'].isin(train_slides_df['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides_df['original_sample'])]
    
    # Calcolo dei pesi delle classi sul training set
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'].to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # --- Augmentations ---
    # Per il training, possiamo aggiungere delle augmentation alla transform base di UNI
    # Nota: Le transform di UNI includono già Resize, ToTensor e Normalize
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        uni_transform # Applica le trasformazioni base di UNI alla fine
    ])
    
    val_transform = uni_transform # Per la validazione, usiamo solo le trasformazioni base
    
    # --- Create Datasets & DataLoaders ---
    train_ds = MILDataset(train_df, train_val_dir, transform=train_transform)
    val_ds = MILDataset(val_df, train_val_dir, transform=val_transform)

    num_workers = min(os.cpu_count() or 2, 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=mil_collate_fn)
    
    input_shape = None # Non rilevante per il modello MIL in questo modo
    
    return train_loader, val_loader, input_shape, class_weights


def get_mil_test_loader(base_path: Path, batch_size: int, uni_transform):
    """
    Crea e restituisce il DataLoader per il test set per il modello MIL.
    """
    test_dir = base_path / "test"
    csv_path = test_dir / "test_patches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV file not found in '{csv_path}'.")
    df = pd.read_csv(csv_path)
    
    # Per il test, usiamo solo le trasformazioni base senza augmentation
    test_transform = uni_transform
    
    test_ds = MILDataset(df, test_dir, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 2, 4), collate_fn=mil_collate_fn)
    
    return test_loader

# =============================================================================
# --- 3. PIPELINE DI VERIFICA (Opzionale, solo per debug) ---
# =============================================================================
# ... La funzione verify_and_visualize può essere aggiunta qui se necessario,
#     ma diventa più complessa da visualizzare per il MIL.
#     È meglio concentrarsi sulle metriche di training.
# =============================================================================

if __name__ == '__main__':
    # Questo blocco serve solo a testare lo script in modo indipendente
    print("Questo script contiene le funzioni per creare i DataLoader MIL e non è pensato per essere eseguito direttamente.")
    print("Importalo nel tuo notebook di training o di inferenza.")