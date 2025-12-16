import cv2
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import zipfile

from pathlib import Path
import cv2
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import zipfile

def zip_directory(folder_path: Path, zip_path: Path):
    print(f"\n>>> Zipping output directory to '{zip_path}'...")
    files_to_zip = [entry for entry in folder_path.rglob('*') if entry.is_file()]
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for entry in tqdm(files_to_zip, desc="Zipping files"):
            zipf.write(entry, entry.relative_to(folder_path))
    print(f"✅ Zipping complete.")

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


# =============================================================================
# --- 3. CONTROLLO QUALITÀ SULL'INTERA SLIDE (Le tue funzioni) ---
# =============================================================================
def contains_slime(img_bgr, threshold=50):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.countNonZero(mask_slime) > threshold

def analyze_image_memory(img_bgr):
    if img_bgr is None: return "FAIL"
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_foreground = (hsv[:,:,1] > 15) & (hsv[:,:,2] < 250)
    foreground_pixels = np.count_nonzero(mask_foreground)
    if foreground_pixels < 100: return "SAFE"
    h_foreground = hsv[:,:,0][mask_foreground]
    count_tissue = np.count_nonzero((h_foreground >= 125) & (h_foreground <= 175))
    count_ink = np.count_nonzero((h_foreground >= 80) & (h_foreground < 125))
    count_shrek_skin = np.count_nonzero((h_foreground >= 20) & (h_foreground < 80))
    count_shrek_clothes = np.count_nonzero((h_foreground >= 10) & (h_foreground < 20))
    count_shrek_total = count_shrek_skin + count_shrek_clothes
    if count_tissue == 0: return "SHREK" if count_shrek_total > 0 else "SAFE"
    ratio_tissue, ratio_shrek = count_tissue/foreground_pixels, count_shrek_total/foreground_pixels
    shrek_dominance = count_shrek_total / count_tissue
    if (count_ink/foreground_pixels) > ratio_shrek and (count_ink/foreground_pixels) > 0.1: return "SAFE"
    if ratio_shrek > 0.4 and shrek_dominance > 4.0: return "SHREK"
    if ratio_tissue > 0.05: return "SAFE"
    if ratio_shrek > 0.3: return "SHREK"
    return "SAFE"

# =============================================================================
# --- 4. MOTORE DI TILING (Invariato dalla versione multi-modale) ---
# =============================================================================
def process_single_slide_multimodal(img_bgr, roi_mask, label, img_path, output_dirs, is_test_set=False):
    # ... (questa funzione rimane identica, dato che il QC è ora nel loop principale)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, tissue_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    img_masked = img_bgr.copy()
    img_masked[tissue_mask == 0] = 255
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles_data = []
    base_name = img_path.stem
    if not contours: return None
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < MIN_ROI_AREA: continue
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        center_x, center_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        tile_name_base = f"{base_name}_roi{i}"
        for scale_name, tile_size in TILE_SIZES.items():
            half_tile = tile_size // 2
            y_start, y_end = center_y - half_tile, center_y + half_tile
            x_start, x_end = center_x - half_tile, center_x + half_tile
            img_crop = img_masked[max(0, y_start):min(img_masked.shape[0], y_end), max(0, x_start):min(img_masked.shape[1], x_end)]
            mask_crop = roi_mask[max(0, y_start):min(roi_mask.shape[0], y_end), max(0, x_start):min(roi_mask.shape[1], x_end)]
            pad_top, pad_bottom = max(0, -y_start), max(0, y_end - img_masked.shape[0])
            pad_left, pad_right = max(0, -x_start), max(0, x_end - img_masked.shape[1])
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                img_crop = cv2.copyMakeBorder(img_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                mask_crop = cv2.copyMakeBorder(mask_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0])
            tile_name = f"{tile_name_base}.png"
            cv2.imwrite(str(output_dirs[f'images_{scale_name}'] / tile_name), img_crop)
            cv2.imwrite(str(output_dirs[f'masks_{scale_name}'] / tile_name), mask_crop)
        row = {'sample_index': f"{tile_name_base}.png", 'original_sample': img_path.name}
        if not is_test_set: row['label'] = label
        tiles_data.append(row)
    return tiles_data

# =============================================================================
# --- 5. WORKFLOW PRINCIPALE (CON QC REINTEGRATO CORRETTAMENTE) ---
# =============================================================================
def run_pipeline(train_dir: Path, labels_csv: Path, output_dir: Path, test_dir: Path = None):
    if output_dir.exists(): shutil.rmtree(output_dir)

    # --- Process TRAINING SET ---
    output_dirs_train = { f"images_{name}": output_dir / f"train/images_{name}" for name in TILE_SIZES.keys() }
    output_dirs_train.update({ f"masks_{name}": output_dir / f"train/masks_{name}" for name in TILE_SIZES.keys() })
    for d in output_dirs_train.values(): d.mkdir(parents=True, exist_ok=True)

    print("\n>>> FASE 1: Processing TRAINING SET (Multi-Modale con QC)")
    if not (train_dir.exists() and labels_csv.exists()):
        print(f"❌ ERROR: Training directory o labels CSV non trovati. Interruzione.")
        return

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

        # --- CONTROLLO QUALITÀ SULL'INTERA SLIDE ---
        if contains_slime(img_bgr) or analyze_image_memory(img_bgr) == "SHREK":
            # Stampa un messaggio opzionale per il debug
            # print(f"\nINFO: Skipping {fname} (QC Failed)")
            continue
        # --- FINE CONTROLLO QUALITÀ ---

        tiles = process_single_slide_multimodal(img_bgr, roi_mask, label, img_path, output_dirs_train, is_test_set=False)
        if tiles: all_train_tiles.extend(tiles)

    if all_train_tiles:
        train_df = pd.DataFrame(all_train_tiles)
        out_train_csv = output_dir / "train/train_patches.csv"
        train_df.to_csv(out_train_csv, index=False)
        print(f"✅ Training Set Completo. Salvati {len(all_train_tiles)} multimodal tile sets.")

    # --- Process TEST SET (logica completa inclusa) ---
    if test_dir and test_dir.exists():
        output_dirs_test = { f"images_{name}": output_dir / f"test/images_{name}" for name in TILE_SIZES.keys() }
        output_dirs_test.update({ f"masks_{name}": output_dir / f"test/masks_{name}" for name in TILE_SIZES.keys() })
        for d in output_dirs_test.values(): d.mkdir(parents=True, exist_ok=True)

        print("\n>>> FASE 2: Processing TEST SET (Multi-Modale)")
        all_test_tiles = []
        test_images = sorted([p for p in test_dir.rglob("img_*.*") if "mask" not in p.name])

        for img_path in tqdm(test_images, desc="Test Slides"):
            id_part = img_path.stem.replace("img_", "")
            mask_path = img_path.parent / f"mask_{id_part}{img_path.suffix}"
            if not mask_path.exists(): mask_path = mask_path.with_suffix('.png')
            if not mask_path.exists(): continue

            img_bgr = load_image_cv2(img_path)
            roi_mask = load_mask_cv2(mask_path)
            if img_bgr is None or roi_mask is None: continue

            # NOTA: il QC non viene applicato al test set
            tiles = process_single_slide_multimodal(img_bgr, roi_mask, None, img_path, output_dirs_test, is_test_set=True)
            if tiles: all_test_tiles.extend(tiles)

        if all_test_tiles:
            test_df = pd.DataFrame(all_test_tiles)
            out_test_csv = output_dir / "test/test_patches.csv"
            test_df.to_csv(out_test_csv, index=False)
            print(f"✅ Test Set Completo. Salvati {len(all_test_tiles)} multimodal tile sets.")
    else:
        print("\nℹ️ Nessuna directory di test fornita o trovata, salto il processing del test set.")

# =============================================================================
# --- 5. ENTRY POINT (Invariato) ---
# =============================================================================
if __name__ == "__main__":
    BASE_PATH = Path('/content/drive/MyDrive/AN2DL_Challenge2-TheBigBatchTheory/data')
    TRAIN_DATA_DIR = BASE_PATH / 'dataset/train_data'
    TEST_DATA_DIR = BASE_PATH / 'dataset/test_data'
    LABELS_CSV_PATH = BASE_PATH / 'dataset/train_labels.csv'
    OUTPUT_PREPROCESSED_DIR = BASE_PATH / 'preprocessed/preprocessed_MaskTile'
    OUTPUT_ZIP_PATH = BASE_PATH / 'preprocessed/preprocessed_MaskTile.zip'

    run_pipeline(
        train_dir=TRAIN_DATA_DIR,
        labels_csv=LABELS_CSV_PATH,
        output_dir=OUTPUT_PREPROCESSED_DIR,
        test_dir=TEST_DATA_DIR
    )
    zip_directory(OUTPUT_PREPROCESSED_DIR, OUTPUT_ZIP_PATH)
