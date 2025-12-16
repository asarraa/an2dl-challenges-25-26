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
# --- 1. DATASET CLASS MULTI-MODALE (IMMAGINE + MASCHERA) ---
# =============================================================================

class MultiModalTileDataset(Dataset):
    """
    Dataset multi-modale che carica una coppia di tile (contesto, dettaglio)
    e le loro maschere ROI corrispondenti, combinandole in un input a 4 canali.
    Restituisce 3 elementi: (img_contesto_4ch, img_dettaglio_4ch, label/nome_file).
    """
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_test = is_test

        # --- MODIFICA: Normalizzazione per 4 canali (RGB + Maschera) ---
        # Usiamo medie e deviazioni standard di ImageNet per RGB e 0.5 per la maschera
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.5]
        )

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row['sample_index']
        
        # --- MODIFICA: Definisci i percorsi per tutte e 4 le immagini ---
        context_img_path = self.base_dir / "images_context" / img_name
        context_mask_path = self.base_dir / "masks_context" / img_name
        detail_img_path = self.base_dir / "images_detail" / img_name
        detail_mask_path = self.base_dir / "masks_detail" / img_name
        
        try:
            img_context = Image.open(context_img_path).convert("RGB")
            mask_context = Image.open(context_mask_path).convert("L") # 'L' per scala di grigi a 8-bit
            img_detail = Image.open(detail_img_path).convert("RGB")
            mask_detail = Image.open(detail_mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Could not read image/mask pair for {img_name}. Error: {e}")

        # --- MODIFICA: Applica le augmentation a immagine e maschera insieme ---
        if self.transform:
            # Per applicare le stesse trasformazioni geometriche (es. flip, rotazione)
            # a immagine e maschera, le "impiliamo" temporaneamente, trasformiamo, e poi separiamo.
            # Convertiamo prima in tensori per poter fare lo stack
            img_context_t = transforms.functional.to_tensor(img_context)
            mask_context_t = transforms.functional.to_tensor(mask_context)
            
            # Stack: [4, H, W] -> [C_img+C_mask, H, W]
            stacked_context = torch.cat([img_context_t, mask_context_t], dim=0)
            
            # Applica le augmentation
            transformed_stacked_context = self.transform(stacked_context)
            
            # Separa di nuovo
            img_context_t = transformed_stacked_context[:3, :, :]
            mask_context_t = transformed_stacked_context[3:, :, :]

            # Fai lo stesso per l'immagine di dettaglio
            img_detail_t = transforms.functional.to_tensor(img_detail)
            mask_detail_t = transforms.functional.to_tensor(mask_detail)
            stacked_detail = torch.cat([img_detail_t, mask_detail_t], dim=0)
            transformed_stacked_detail = self.transform(stacked_detail)
            img_detail_t = transformed_stacked_detail[:3, :, :]
            mask_detail_t = transformed_stacked_detail[3:, :, :]
        else:
            # Se non ci sono augmentation, converti solo in tensori
            img_context_t = transforms.functional.to_tensor(img_context)
            mask_context_t = transforms.functional.to_tensor(mask_context)
            img_detail_t = transforms.functional.to_tensor(img_detail)
            mask_detail_t = transforms.functional.to_tensor(mask_detail)

        # --- MODIFICA: Combina immagine e maschera in un tensore a 4 canali e normalizza ---
        final_context = self.normalize(torch.cat([img_context_t, mask_context_t], dim=0))
        final_detail = self.normalize(torch.cat([img_detail_t, mask_detail_t], dim=0))
        
        if self.is_test:
            return final_context, final_detail, img_name
        else:
            label = row['label']
            return final_context, final_detail, torch.tensor(label, dtype=torch.long)

# =============================================================================
# --- 2. FUNZIONI PER CREARE I DATALOADER (AGGIORNATE) ---
# =============================================================================
def get_multimodal_loaders(base_path: Path, batch_size: int, val_split: float = 0.2, seed: int = 42):
    print("\n--- Creating Multi-Modal (Image+Mask) Train & Validation DataLoaders ---")
    train_val_dir = base_path / "train"
    csv_path = train_val_dir / "train_patches.csv"
    # ... (la logica di split e mappatura delle label rimane la stessa)
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df['label'].unique())
    label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    unique_slides_df = df.groupby('original_sample')['label'].first().reset_index()
    train_slides, val_slides = train_test_split(unique_slides_df, test_size=val_split, random_state=seed, stratify=unique_slides_df['label'])
    train_df = df[df['original_sample'].isin(train_slides['original_sample'])]
    val_df = df[df['original_sample'].isin(val_slides['original_sample'])]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'].to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # --- MODIFICA: Le augmentation ora lavorano su tensori a 4 canali ---
    train_augmentation = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Applichiamo ColorJitter solo ai canali RGB
        transforms.Lambda(lambda x: torch.cat([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)(x[:3,:,:]),
            x[3:,:,:] # Lascia il canale della maschera invariato
        ], dim=0)),
    ])

    val_augmentation = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])
    
    train_ds = MultiModalTileDataset(train_df, train_val_dir, transform=train_augmentation)
    val_ds = MultiModalTileDataset(val_df, train_val_dir, transform=val_augmentation)

    num_workers = min(os.cpu_count() or 2, 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # --- MODIFICA: Input shape ora ha 4 canali ---
    input_shape = (4, 224, 224)
    
    return train_loader, val_loader, input_shape, class_weights

def get_multimodal_test_loader(base_path: Path, batch_size: int):
    print("\n--- Creating Multi-Modal Test DataLoader ---")
    test_dir = base_path / "test"
    csv_path = test_dir / "test_patches.csv"
    df = pd.read_csv(csv_path)
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])
    
    test_ds = MultiModalTileDataset(df, test_dir, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 2, 8), pin_memory=True)
    
    input_shape = (4, 224, 224)
    
    return test_loader, input_shape

# =============================================================================
# --- 3. PIPELINE DI VERIFICA (AGGIORNATA) ---
# =============================================================================
def verify_and_visualize(loader: DataLoader, num_images: int = 8, dataset_name: str = "Dataset"):
    print(f"\n--- Verifying and Visualizing a batch from the {dataset_name} ---")
    context_batch, detail_batch, labels_or_names = next(iter(loader))
    
    print(f"  Context Batch Shape: {context_batch.shape}") # Should be [B, 4, 224, 224]
    print(f"  Detail Batch Shape: {detail_batch.shape}")   # Should be [B, 4, 224, 224]

    mean, std = np.array([0.485, 0.456, 0.406, 0.5]), np.array([0.229, 0.224, 0.225, 0.5])
    
    fig, axs = plt.subplots(3, num_images, figsize=(20, 7)) # Aggiunta una riga per la maschera
    fig.suptitle(f"Sample Batch from {dataset_name} (Image + Mask)", fontsize=16)

    for i in range(min(num_images, len(context_batch))):
        # --- Immagine di Contesto (RGB) ---
        img_ctx = context_batch[i][:3,:,:].numpy().transpose((1, 2, 0)); 
        img_ctx = std[:3] * img_ctx + mean[:3]; img_ctx = np.clip(img_ctx, 0, 1)
        axs[0, i].imshow(img_ctx)
        axs[0, i].set_title(f"Context\nLabel: {labels_or_names[i].item() if isinstance(labels_or_names, torch.Tensor) else 'N/A'}")
        axs[0, i].axis("off")
        
        # --- Maschera di Contesto ---
        mask_ctx = context_batch[i][3,:,:].numpy() # Canale 4
        axs[1, i].imshow(mask_ctx, cmap='gray')
        axs[1, i].set_title(f"Context Mask")
        axs[1, i].axis("off")

        # --- Immagine di Dettaglio (RGB) ---
        img_det = detail_batch[i][:3,:,:].numpy().transpose((1, 2, 0)); 
        img_det = std[:3] * img_det + mean[:3]; img_det = np.clip(img_det, 0, 1)
        axs[2, i].imshow(img_det)
        axs[2, i].set_title(f"Detail")
        axs[2, i].axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# --- 4. ENTRY POINT (Invariato) ---
# =============================================================================
if __name__ == '__main__':
    # ... (la logica di argparse rimane la stessa)
    pass