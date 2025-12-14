import cv2
import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
# --- FIX IMPORT: Usiamo la versione stabile standard, non la v2 ---
from torchvision import transforms 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import config
from sklearn.utils.class_weight import compute_class_weight

# Global random seed for reproducibility
SEED = 42

LOADER_PARAMS = config.LOADER_PARAMS

class LazyImageDataset(torch.utils.data.Dataset):
    """
    Memory-efficient Dataset that loads images from disk on-demand.
    """
    def __init__(self, csv_df, images_dir, masks_dir=None, add_mask_channel=False, transform=None):
        self.resize = transforms.Resize((224, 224))
        self.csv_df = csv_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.add_mask_channel = add_mask_channel
        self.transform = transform
        
        # 1. Definizione statistiche Normalizzazione
        if self.add_mask_channel:
            # RGB + Mask
            self.mean = [0.485, 0.456, 0.406, 0.5]
            self.std  = [0.229, 0.224, 0.225, 0.5]
        else:
            # Solo RGB
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]

        # 2. Base transform: Conversione sicura Numpy [0,255] -> Tensor [0,1]
        self.to_tensor_base = transforms.ToTensor()

        # 3. Normalizzazione finale
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        row = self.csv_df.iloc[idx]
        img_name = row['sample_index']
        label = row['label']
        
        # --- CARICAMENTO ---
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- GESTIONE MASCHERA ---
        if self.add_mask_channel:
            if self.masks_dir is None:
                raise ValueError("masks_dir must be provided when add_mask_channel=True")
            
            mask_path = self.masks_dir / img_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise FileNotFoundError(f"Could not load mask {mask_path}")
            
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            image = np.dstack((image, mask))
        
        # --- CHECK PATCH NERI (Debug) ---
        if np.mean(image) < 1.0: 
            print(f"⚠️ WARNING: Patch quasi nero rilevato! {img_name} (Mean: {np.mean(image):.2f})")

        # --- PIPELINE TRASFORMAZIONI ---
        
        # 1. ToTensor ([0, 1])
        image_tensor = self.to_tensor_base(image)
        image_tensor = self.resize(image_tensor)
        
        # 2. Augmentations
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        # 3. Normalize ([-2, +2])
        image_tensor = self.normalize(image_tensor)
        
        # --- DEBUG PRIMA IMMAGINE ---
        if idx == 0:
            print(f"Tensor Stats for {img_name}:")
            print(f"   Min: {image_tensor.min():.2f} (Target: negativo)")
            print(f"   Max: {image_tensor.max():.2f} (Target: positivo)")
            print(f"   Mean: {image_tensor.mean():.2f} (Target: ~0)")
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


class TestImageDataset(torch.utils.data.Dataset):
    """
    Dataset for inference. FIX: Aggiunta Normalizzazione uguale al Training.
    """
    def __init__(self, filenames, images_dir, masks_dir=None, add_mask_channel=False):
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.add_mask_channel = add_mask_channel
        
        # --- FIX: COPIA LOGICA NORMALIZZAZIONE DAL TRAINING ---
        if self.add_mask_channel:
            self.mean = [0.485, 0.456, 0.406, 0.5]
            self.std  = [0.229, 0.224, 0.225, 0.5]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]

        self.to_tensor_base = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # -------------------------------------------------------

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.add_mask_channel:
            if self.masks_dir is None:
                raise ValueError("masks_dir must be provided when add_mask_channel=True")
            mask_path = self.masks_dir / img_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load mask {mask_path}")
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image = np.dstack((image, mask))

        # --- FIX: PIPELINE CORRETTA ---
        image_tensor = self.to_tensor_base(image) # [0, 1]
        image_tensor = self.normalize(image_tensor) # [-2, 2]
        # ------------------------------

        return image_tensor, img_name

def make_loader(ds, batch_size, drop_last, shuffle=False, sampler=None):
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_cores))
    
    # Se c'è sampler, shuffle deve essere False
    if sampler is not None:
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4,
        persistent_workers=True,
    )

def _resolve_paths(base_path, add_mask_channel=False, is_test=False):
    split = "test" if is_test else "train"
    if base_path is not None:
        root = Path(base_path)
        images_dir = root / split / "images"
        masks_dir = root / split / "masks" if add_mask_channel else None
        csv_name = "test_patches.csv" if is_test else "train_patches.csv"
        csv_path = root / split / csv_name
    else:
        split_dir = config.TEST_DIR if is_test else config.TRAIN_DIR
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks" if add_mask_channel else None
        csv_name = "test_patches.csv" if is_test else "train_patches.csv"
        csv_path = split_dir / split / csv_name
    return images_dir, masks_dir, csv_path

def get_loaders(augmentation=None, batch_size=LOADER_PARAMS["batch_size"], base_path=None, add_mask_channel=False, use_sampler=False):
    images_dir, masks_dir, csv_path = _resolve_paths(base_path, add_mask_channel, is_test=False)
    
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].map(config.LABEL_MAP)
    
    unique_img_df = df.groupby('original_sample')['label'].first().reset_index()
    
    train_imgs, val_imgs = train_test_split(
        unique_img_df['original_sample'],
        test_size=LOADER_PARAMS["percentage_validation"],
        random_state=SEED,
        stratify=unique_img_df['label']
    )
    
    train_df = df[df['original_sample'].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df['original_sample'].isin(val_imgs)].reset_index(drop=True)
    
    sample_path = images_dir / df.iloc[0]['sample_index']
    sample_img = cv2.imread(str(sample_path))
    if sample_img is None:
        raise FileNotFoundError(f"Could not load sample image {sample_path}")
     
    # --- CALCOLO PESI CORRETTO ---
    all_labels = df['label'].to_numpy()
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32)
    print("Class Weights calcolati (sui patch):", class_weights)
    
    if use_sampler:
        train_labels = train_df['label'].to_numpy()
        
        # B. Calcola pesi delle classi basandoti solo sul training (più sicuro)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        # Converti in tensore (utile per verifica, non critico per il sampler che vuole lista)
        class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float32)
        print("   Class Weights:", class_weights_tensor)
        
        # C. Assegna il peso a ogni singolo campione del TRAIN
        #    IMPORTANTE: Usiamo train_labels, non all_labels!
        sample_weights = [class_weights_tensor[label] for label in train_labels]
        
        # D. Crea il Sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    # Input Shape Logic
    if add_mask_channel:
        sample_mask = cv2.imread(str(masks_dir / df.iloc[0]['sample_index']), cv2.IMREAD_GRAYSCALE)
        if sample_mask is None:
            raise FileNotFoundError(f"Could not load sample mask")
        input_shape = (4, sample_img.shape[0], sample_img.shape[1])
    else:
        input_shape = (3, sample_img.shape[0], sample_img.shape[1])
    
    # Augmentations (Aggressive)
    if augmentation is None:
        train_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=90),
        ])
    else:
        train_augmentation = augmentation
    
    train_ds = LazyImageDataset(train_df, images_dir, masks_dir=masks_dir, add_mask_channel=add_mask_channel, transform=train_augmentation)
    val_ds = LazyImageDataset(val_df, images_dir, masks_dir=masks_dir, add_mask_channel=add_mask_channel, transform=None)
    
    if use_sampler:
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=False)
    else:
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, sampler=None, drop_last=False)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, sampler=None, drop_last=False)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Input shape: {input_shape}")
    
    return train_loader, val_loader, input_shape, class_weights

def get_test_loaders(add_mask_channel=False, batch_size=LOADER_PARAMS["batch_size"], base_path=None):
    images_dir, masks_dir, csv_path = _resolve_paths(base_path, add_mask_channel, is_test=True)

    df = pd.read_csv(csv_path)
    # Mapping labels non necessario per il test set se non ci sono label, ma se è test interno con label, serve.
    # Qui assumiamo inference pura (niente label), quindi niente map.
    
    sample_path = images_dir / df.iloc[0]['sample_index']
    sample_img = cv2.imread(str(sample_path))
    
    if add_mask_channel:
        input_shape = (4, sample_img.shape[0], sample_img.shape[1])
    else:
        input_shape = (3, sample_img.shape[0], sample_img.shape[1])
    
    ds = TestImageDataset(
        filenames=df['sample_index'],
        images_dir=images_dir,
        masks_dir=masks_dir,
        add_mask_channel=add_mask_channel
    )
    
    test_loader = make_loader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Test samples: {len(ds)}, Input shape: {input_shape}")
    
    return test_loader, input_shape