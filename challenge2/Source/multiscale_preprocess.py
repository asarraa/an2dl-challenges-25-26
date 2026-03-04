#!/usr/bin/env python3

import cv2
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import zipfile

# =============================================================================
# --- 1. CONFIGURAZIONE GLOBALE ---
# =============================================================================
TILE_SIZES = {
    'context': 768,
    'detail': 256
}
MIN_ROI_AREA = 100

# =============================================================================
# --- 2. FUNZIONI DI UTILITÀ E I/O (Invariate) ---
# =============================================================================
def load_image_cv2(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def load_mask_cv2(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def zip_directory(folder_path: Path, zip_path: Path):
    print(f"\n>>> Zipping output directory to '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for entry in folder_path.rglob('*'):
            zipf.write(entry, entry.relative_to(folder_path))
    print(f"✅ Zipping complete.")

# =============================================================================
# --- 3. MOTORE DI TILING (AGGIORNATO PER IMMAGINE + MASCHERA) ---
# =============================================================================
def process_single_slide_multimodal(img_bgr, roi_mask, label, img_path, output_dirs, is_test_set=False):
    # Crea un'immagine con sfondo bianco per la visualizzazione e il training
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, tissue_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    img_masked = img_bgr.copy()
    img_masked[tissue_mask == 0] = 255

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tiles_data = []
    base_name = img_path.stem
    
    if not contours: return None

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < MIN_ROI_AREA:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        tile_name_base = f"{base_name}_roi{i}"

        # Estrai un tile per ogni scala definita
        for scale_name, tile_size in TILE_SIZES.items():
            half_tile = tile_size // 2
            y_start, y_end = center_y - half_tile, center_y + half_tile
            x_start, x_end = center_x - half_tile, center_x + half_tile
            
            # --- MODIFICA 1: Estrai il crop sia dall'immagine che dalla maschera ROI ---
            img_crop = img_masked[max(0, y_start):min(img_masked.shape[0], y_end), max(0, x_start):min(img_masked.shape[1], x_end)]
            mask_crop = roi_mask[max(0, y_start):min(roi_mask.shape[0], y_end), max(0, x_start):min(roi_mask.shape[1], x_end)]

            # Calcola il padding necessario
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - img_masked.shape[0])
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - img_masked.shape[1])
            
            # Applica il padding sia all'immagine che alla maschera
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                img_crop = cv2.copyMakeBorder(img_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                mask_crop = cv2.copyMakeBorder(mask_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0]) # Pad della maschera con nero (0)
            
            # --- MODIFICA 2: Salva sia il tile dell'immagine che il tile della maschera ---
            tile_name = f"{tile_name_base}.png"
            
            # Salva l'immagine nella sua cartella
            output_dir_img = output_dirs[f'images_{scale_name}']
            cv2.imwrite(str(output_dir_img / tile_name), img_crop)

            # Salva la maschera nella sua cartella
            output_dir_mask = output_dirs[f'masks_{scale_name}']
            cv2.imwrite(str(output_dir_mask / tile_name), mask_crop)

        # Aggiungi una sola riga al CSV per ogni coppia di tile (img+mask)
        row = {'sample_index': f"{tile_name_base}.png", 'original_sample': img_path.name}
        if not is_test_set:
            row['label'] = label
        tiles_data.append(row)
            
    return tiles_data

# =============================================================================
# --- 4. WORKFLOW PRINCIPALE (AGGIORNATO) ---
# =============================================================================
def run_pipeline(train_dir: Path, labels_csv: Path, output_dir: Path, test_dir: Path = None):
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    # --- MODIFICA 3: Crea cartelle separate per immagini e maschere ---
    output_dirs_train = {
        f"images_{name}": output_dir / f"train/images_{name}" for name in TILE_SIZES.keys()
    }
    output_dirs_train.update({
        f"masks_{name}": output_dir / f"train/masks_{name}" for name in TILE_SIZES.keys()
    })
    for d in output_dirs_train.values(): d.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> FASE 1: Processing TRAINING SET (Multi-Modale: Immagine + Maschera)")
    # (La logica del loop principale rimane la stessa, ma ora chiamiamo la nuova funzione)
    labels_df = pd.read_csv(labels_csv)
    all_train_tiles = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Train Slides"):
        fname, label = row['sample_index'], row['label']
        # ... (la logica per trovare i file rimane la stessa)
        img_paths = list(train_dir.glob(f"**/{fname}"))
        if not img_paths: continue
        img_path = img_paths[0]
        mask_path = img_path.parent / fname.replace("img_", "mask_")
        if not mask_path.exists(): mask_path = mask_path.with_suffix('.png')
        if not mask_path.exists(): continue
        
        img_bgr = load_image_cv2(img_path)
        roi_mask = load_mask_cv2(mask_path)
        if img_bgr is None or roi_mask is None: continue
        
        # (La logica di QC può essere aggiunta qui se necessario)

        tiles = process_single_slide_multimodal(img_bgr, roi_mask, label, img_path, output_dirs_train, is_test_set=False)
        if tiles: all_train_tiles.extend(tiles)

    if all_train_tiles:
        train_df = pd.DataFrame(all_train_tiles)
        out_train_csv = output_dir / "train/train_patches.csv"
        train_df.to_csv(out_train_csv, index=False)
        print(f"✅ Training Set Complete. Saved {len(all_train_tiles)} multimodal tile sets.")

    # (La logica per il test set va aggiornata in modo simile)
    # ...

# =============================================================================
# --- 5. ENTRY POINT (Invariato) ---
# =============================================================================
if __name__ == "__main__":
    # La logica di argparse e Colab rimane la stessa
    # ...
    pass