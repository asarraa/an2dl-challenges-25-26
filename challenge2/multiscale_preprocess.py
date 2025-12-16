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
# Definiamo le due scale che vogliamo estrarre
TILE_SIZES = {
    'context': 768,  # Per il contesto architettonico (vista "lontana")
    'detail': 256    # Per i dettagli cellulari (vista "vicina")
}
MIN_ROI_AREA = 100 # Area minima in pixel per considerare un ROI valido

# =============================================================================
# --- 2. FUNZIONI DI UTILITÀ E I/O ---
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
# --- 3. MOTORE DI TILING (AGGIORNATO PER MULTI-SCALA) ---
# =============================================================================
def process_single_slide_multiscale(img_bgr, roi_mask, label, img_path, output_dirs, is_test_set=False):
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
            
            img_crop = img_masked[max(0, y_start):min(img_masked.shape[0], y_end), max(0, x_start):min(img_masked.shape[1], x_end)]

            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - img_masked.shape[0])
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - img_masked.shape[1])
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                img_crop = cv2.copyMakeBorder(img_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # Salva il tile nella sua cartella specifica
            tile_name = f"{tile_name_base}.png"
            output_dir = output_dirs[scale_name]
            cv2.imwrite(str(output_dir / tile_name), img_crop)

        # Aggiungi una sola riga al CSV per entrambi i tile
        row = {'sample_index': f"{tile_name_base}.png", 'original_sample': img_path.name}
        if not is_test_set:
            row['label'] = label
        tiles_data.append(row)
            
    return tiles_data

# =============================================================================
# --- 4. WORKFLOW PRINCIPALE ---
# =============================================================================
def run_pipeline(train_dir: Path, labels_csv: Path, output_dir: Path, test_dir: Path = None):
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    # Crea sottocartelle per ogni scala
    output_dirs_train = {name: output_dir / f"train/images_{name}" for name in TILE_SIZES}
    for d in output_dirs_train.values(): d.mkdir(parents=True, exist_ok=True)
    
    print("\n>>> FASE 1: Processing TRAINING SET (Multi-Scala)")
    labels_df = pd.read_csv(labels_csv)
    all_train_tiles = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Train Slides"):
        fname, label = row['sample_index'], row['label']
        img_paths = list(train_dir.glob(f"**/{fname}"))
        if not img_paths: continue
        img_path = img_paths[0]
        mask_path = img_path.parent / fname.replace("img_", "mask_")
        if not mask_path.exists(): mask_path = mask_path.with_suffix('.png')
        if not mask_path.exists(): continue
        
        img_bgr = load_image_cv2(img_path)
        roi_mask = load_mask_cv2(mask_path)
        if img_bgr is None or roi_mask is None: continue

        tiles = process_single_slide_multiscale(img_bgr, roi_mask, label, img_path, output_dirs_train, is_test_set=False)
        if tiles: all_train_tiles.extend(tiles)

    if all_train_tiles:
        train_df = pd.DataFrame(all_train_tiles)
        out_train_csv = output_dir / "train/train_patches.csv"
        train_df.to_csv(out_train_csv, index=False)
        print(f"✅ Training Set Complete. Saved {len(all_train_tiles)} tile pairs.")

    # (La logica per il test set è simile e può essere aggiunta se necessario)

# =============================================================================
# --- 5. ENTRY POINT ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale ROI-centric tiling preprocessor.")
    # (Argparse logic rimane la stessa, qui definisco i percorsi per Colab)

    from google.colab import drive
    drive.mount('/content/drive')
    
    BASE_PATH = Path('/content/drive/MyDrive/KaggleCompetition/')
    TRAIN_DATA_DIR = BASE_PATH / 'train_data'
    LABELS_CSV_PATH = BASE_PATH / 'train_labels.csv'
    
    OUTPUT_PREPROCESSED_DIR = Path('/content/multiscale_preprocessed_data')
    OUTPUT_ZIP_PATH = BASE_PATH / 'multiscale_preprocessed_data.zip'
    
    run_pipeline(
        train_dir=TRAIN_DATA_DIR,
        labels_csv=LABELS_CSV_PATH,
        output_dir=OUTPUT_PREPROCESSED_DIR,
    )
    zip_directory(OUTPUT_PREPROCESSED_DIR, OUTPUT_ZIP_PATH)