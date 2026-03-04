import cv2
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import config

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# MODIFICA: Aumentato a 512 per catturare più contesto architettonico.
# La rete poi farà resize a 224, vedendo "più roba" anche se meno dettagliata.
TILE_SIZE = 224

# Stride per l'overlap (metà del tile size per avere buon TTA)
STRIDE = 112  # 50% overlap

# Soglia minima di tessuto utile (training).
TISSUE_THRESHOLD_RATIO = 0.15  # 15% di tessuto minimo
# Soglia minima per il test: teniamo quasi tutto, ma scartiamo <1% di tessuto.
TEST_TISSUE_THRESHOLD_RATIO = 0.01
# Peso minimo per mantenere rilevanza anche ai patch con poco tessuto nel test.
MIN_PATCH_WEIGHT = 0.01

def compute_patch_weight(tissue_ratio: float) -> float:
    """
    Restituisce un peso crescente con il contenuto di tessuto.
    Un floor evita pesi nulli quando il test set viene mantenuto integralmente.
    """
    ratio = float(tissue_ratio)
    return min(1.0, max(MIN_PATCH_WEIGHT, ratio))

# =============================================================================
# 1. UTILITY & I/O FUNCTIONS
# =============================================================================

def load_image_cv2(path):
    # Load in BGR
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def load_mask_cv2(path):
    # Load in Grayscale
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# =============================================================================
# 2. PREPROCESSING LOGIC (SLIME REMOVAL & ARTIFACTS)
# =============================================================================
def is_outlier(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)
    if cv2.countNonZero(mask_slime) > 50:
        return True
    
    mask_foreground = (hsv[:,:,1] > 15) & (hsv[:,:,2] < 250)
    foreground_pixels = np.count_nonzero(mask_foreground)
    
    if foreground_pixels < 100: return False

    h_foreground = hsv[:,:,0][mask_foreground]

    # Count pixels by Hue range
    count_tissue = np.count_nonzero((h_foreground >= 125) & (h_foreground <= 175))
    count_ink = np.count_nonzero((h_foreground >= 80) & (h_foreground < 125))
    count_shrek_skin = np.count_nonzero((h_foreground >= 20) & (h_foreground < 80))
    count_shrek_clothes = np.count_nonzero((h_foreground >= 10) & (h_foreground < 20))
    count_shrek_total = count_shrek_skin + count_shrek_clothes
    
    ratio_tissue = count_tissue / foreground_pixels
    ratio_ink = count_ink / foreground_pixels
    ratio_shrek = count_shrek_total / foreground_pixels
    shrek_dominance = (count_shrek_total / count_tissue) if count_tissue > 0 else 999.0

    # Rules
    if ratio_ink > ratio_shrek and ratio_ink > 0.1: return False
    if ratio_shrek > 0.4 and shrek_dominance > 4.0: return True
    if ratio_tissue > 0.05: return False
    if ratio_shrek > 0.3: return True

    return False
    

def contains_slime(img_bgr, threshold=50):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)
    return cv2.countNonZero(mask_slime) > threshold

def analyze_image_memory(img_bgr):
    """
    Rileva artefatti 'Shrek' (pelle, vestiti, pennarello grossolano).
    """
    if img_bgr is None: return "SAFE", 0, 0, 0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 
    
    # Foreground approssimativo (non bianco)
    mask_foreground = (hsv[:,:,1] > 15) & (hsv[:,:,2] < 250)
    foreground_pixels = np.count_nonzero(mask_foreground)
    
    if foreground_pixels < 100: return "SAFE (Empty)", 0.0, 0.0, 0.0

    h_foreground = hsv[:,:,0][mask_foreground]

    # Count pixels by Hue range
    count_tissue = np.count_nonzero((h_foreground >= 125) & (h_foreground <= 175))
    count_ink = np.count_nonzero((h_foreground >= 80) & (h_foreground < 125))
    count_shrek_skin = np.count_nonzero((h_foreground >= 20) & (h_foreground < 80))
    count_shrek_clothes = np.count_nonzero((h_foreground >= 10) & (h_foreground < 20))
    count_shrek_total = count_shrek_skin + count_shrek_clothes
    
    ratio_tissue = count_tissue / foreground_pixels
    ratio_ink = count_ink / foreground_pixels
    ratio_shrek = count_shrek_total / foreground_pixels
    shrek_dominance = (count_shrek_total / count_tissue) if count_tissue > 0 else 999.0

    # Rules
    if ratio_ink > ratio_shrek and ratio_ink > 0.1: return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance
    if ratio_shrek > 0.4 and shrek_dominance > 4.0: return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance
    if ratio_tissue > 0.05: return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance
    if ratio_shrek > 0.3: return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance

    return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

# =============================================================================
# 3. TILING ENGINE (IL CORE MODIFICATO)
# =============================================================================

def process_single_slide(img_path, mask_path, label, output_img_dir, is_test_set=False):
    img_bgr = load_image_cv2(img_path)
    mask = load_mask_cv2(mask_path)
    
    if img_bgr is None or mask is None: return None

    # --- Step 1: Quality Control (Solo per training) ---
    if not is_test_set:
        if contains_slime(img_bgr): return
        cls, _, _, _ = analyze_image_memory(img_bgr)
        if cls == "SHREK": return

    # --- Step 2: Applicazione Maschera (Bianco invece che Nero) ---
    # Creiamo un'immagine pulita dove lo sfondo (mask==0) diventa BIANCO (255)
    # Le CNN preferiscono il bianco al nero per lo sfondo in istologia.
    img_masked = img_bgr.copy()
    img_masked[mask == 0] = 255 # Sfondo bianco

    # --- Step 3: Tiling ---
    tiles_data = []
    h, w, _ = img_bgr.shape
    base_name = img_path.stem 

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            
            # Scarta strisce troppo piccole ai bordi
            if (y_end - y) < TILE_SIZE // 3 or (x_end - x) < TILE_SIZE // 3: continue

            # Estrai crop immagine e crop maschera
            img_crop = img_masked[y:y_end, x:x_end]
            mask_crop = mask[y:y_end, x:x_end]

            # --- FILTRO CRITICO: TISSUE RATIO ---
            # Contiamo i pixel validi (che sono 255 nella maschera originale)
            tissue_pixels = cv2.countNonZero(mask_crop)
            total_pixels_crop = mask_crop.shape[0] * mask_crop.shape[1]
            tissue_ratio = tissue_pixels / total_pixels_crop
            
            # Training: scarta patch quasi vuoti.
            if not is_test_set and tissue_ratio < TISSUE_THRESHOLD_RATIO:
                continue
            # Test: scarta solo patch <1% tessuto per non sprecare memoria.
            if is_test_set and tissue_ratio < TEST_TISSUE_THRESHOLD_RATIO:
                continue
            
            # --- Padding (Reflection) ---
            pad_h = TILE_SIZE - img_crop.shape[0]
            pad_w = TILE_SIZE - img_crop.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                img_crop = cv2.copyMakeBorder(img_crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                # Padding su bianco per evitare artefatti neri? 
                # Reflect è meglio perché continua la texture.

            # Calcolo peso crescente con il contenuto di tessuto (train e test).
            weight = compute_patch_weight(tissue_ratio)
                
            tile_name = f"{base_name}_y{y}_x{x}.png"
            cv2.imwrite(str(output_img_dir / tile_name), img_crop)

            row = {
                'sample_index': tile_name,
                'original_sample': img_path.name,
                'tissue_ratio': tissue_ratio, # Salviamo quanto tessuto c'è, utile per analisi
                'weight': weight,
            }
            
            if not is_test_set:
                row['label'] = label
            
            tiles_data.append(row)
            
    return tiles_data

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def preprocess(do_test=True, preprocess_name=None):
    if preprocess_name == None:
        print("[ERROR] Give a name to this preprocessing")
        return

    single_preprocessing_dir = config.BASE_PREPROCESSED / preprocess_name
    out_train_img = single_preprocessing_dir / "train/images"
    out_test_img = single_preprocessing_dir / "test/images"
    
    # Cleanup e Creazione Cartelle
    if out_train_img.exists(): shutil.rmtree(out_train_img)
    out_train_img.mkdir(parents=True, exist_ok=True)

    if do_test:
        if out_test_img.exists(): shutil.rmtree(out_test_img)
        out_test_img.mkdir(parents=True, exist_ok=True)

    # --- TRAINING SET ---
    print(">>> FASE 1: Processing TRAINING SET")
    labels_csv = config.LABELS_CSV
    train_dir = config.TRAIN_DIR
    
    if labels_csv.exists() and train_dir.exists():
        labels_df = pd.read_csv(labels_csv).sort_values(by='sample_index')
        train_rows = []
        
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Train Slides"):
            fname = row['sample_index']
            label = row['label']
            
            img_path = train_dir / fname
            if not img_path.exists(): 
                 found = list(train_dir.glob(f"**/{fname}")) 
                 if found: img_path = found[0]
                 else: continue
            
            mask_name = fname.replace("img_", "mask_")
            mask_path = img_path.parent / mask_name
            # Fallback estensioni
            if not mask_path.exists(): 
                 mask_path = img_path.parent / fname.replace("img_", "mask_").replace(".jpg", ".png")
                 if not mask_path.exists(): continue

            res = process_single_slide(img_path, mask_path, label, out_train_img, is_test_set=False)
            if isinstance(res, list): train_rows.extend(res)

        if train_rows:
            train_df = pd.DataFrame(train_rows)
            # Salviamo anche tissue_ratio per curiosità
            cols = ['sample_index', 'original_sample', 'label', 'tissue_ratio', 'weight']
            train_df = train_df[cols]
            train_df.to_csv(single_preprocessing_dir / "train/train_patches.csv", index=False)
            print(f"✅ Training Tiles Saved: {len(train_rows)}")
        else:
            print("⚠️ No tiles generated for Training Set.")
    else:
        print("⚠️ train_data folder missing.")
        
    if do_test and config.TEST_DIR.exists():
        process_test(preprocess_name)

def process_test(preprocess_name=None):
    single_preprocessing_dir = config.BASE_PREPROCESSED / preprocess_name
    out_test_img = single_preprocessing_dir / "test/images"

    print("\n>>> FASE 2: Processing TEST SET")
    test_dir = config.TEST_DIR
    if test_dir.exists():
        test_rows = []
        all_files = sorted(list(test_dir.glob("**/img_*.*")))
    
        for img_path in tqdm(all_files, desc="Test Slides"):
            if "mask" in img_path.name: continue
            
            id_part = img_path.stem.replace("img_", "")
            mask_path = img_path.parent / f"mask_{id_part}{img_path.suffix}"
            if not mask_path.exists():
                mask_path = img_path.parent / f"mask_{id_part}.png"
            if not mask_path.exists(): continue 

            res = process_single_slide(img_path, mask_path, None, out_test_img, is_test_set=True)
            if isinstance(res, list): test_rows.extend(res)
            
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            cols = ['sample_index', 'original_sample', 'tissue_ratio', 'weight']
            test_df = test_df[cols]
            test_df.to_csv(single_preprocessing_dir / "test/test_patches.csv", index=False)
            print(f"✅ Test Tiles Saved: {len(test_rows)}")
        else:
            print("⚠️ No tiles generated for Test Set.")
    else:
        print("⚠️ test_data folder missing.")

if __name__ == "__main__":
    # Usa un nome nuovo per non sovrascrivere se vuoi comparare
    preprocess(do_test=True, preprocess_name="preprocess_v2_224_white_15%")
