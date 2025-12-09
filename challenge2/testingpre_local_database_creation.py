import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURAZIONE DI DEFAULT (LOCALE)
# =============================================================================

DEFAULT_RAW_DATA = Path("./data")
DEFAULT_OUTPUT_DIR = Path("./data/temp_processed")
DEFAULT_ZIP_DIR = Path("./data/output_zips")

# Parametri Generali Tessuto
# Soglie per capire se è "vetro" o tessuto
HSV_S_THRESH = 15   
HSV_V_THRESH = 200  
BACKGROUND_MAX_RATIO = 0.5 

# =============================================================================
# VARIANTI AGGIORNATE (DIMENSIONI DIVERSE)
# =============================================================================
PREPROCESSING_VARIANTS = {
    # ---------------------------------------------------------
    # 1. VARIANTE "IMAGENET" (224x224) - VELOCISSIMA
    # ---------------------------------------------------------
    "tiny_224_imagenet": {
        "description": "224x224 px. Formato standard per Transfer Learning (ResNet, EfficientNet). Leggerissimo.",
        "tile_size": 224,
        "stride": 224,      # Nessun overlap per tenere il dataset piccolo
        "downscale_factor": 1.0,
        "save_arrays": False,
    },

    # ---------------------------------------------------------
    # 2. VARIANTE "STANDARD SMALL" (256x256) - ALTA RISOLUZIONE
    # ---------------------------------------------------------
    "small_256_native": {
        "description": "256x256 px. Zoom 1:1. Ottimo dettaglio cellulare, ma poco contesto attorno.",
        "tile_size": 256,
        "stride": 256,
        "downscale_factor": 1.0,
        "save_arrays": False,
    },
    
    "small_128_native": {
        "description": "128x128 px. Zoom 1:1. Ottimo dettaglio cellulare, ma poco contesto attorno.",
        "tile_size": 128,
        "stride": 128,
        "downscale_factor": 1.0,
        "save_arrays": False,
    },

    # ---------------------------------------------------------
    # 3. VARIANTE "CONTEXT SMART" (256x256 Downscaled) - IL BEST SELLER?
    # ---------------------------------------------------------
    "context_256_zoomed_out": {
        "description": "256x256 px finali, ma ottenuti rimpicciolendo un'area di 512px. Vedi più contesto in poca memoria.",
        "tile_size": 256,
        "stride": 256,      
        "downscale_factor": 0.5, # Rimpicciolisce l'immagine PRIMA di tagliare
        "save_arrays": False,
    },

    # ---------------------------------------------------------
    # 4. VARIANTE "BIG" (512x512) - PER RETI POTENTI
    # ---------------------------------------------------------
    "standard_512": {
        "description": "512x512 px. Molto dettaglio e molto contesto. Pesante per il training.",
        "tile_size": 512,
        "stride": 512,
        "downscale_factor": 1.0,
        "save_arrays": False,
    },
}

# =============================================================================
# CLASSI & UTILS
# =============================================================================

@dataclass
class PreprocessingConfig:
    name: str
    base_data: Path
    output_root: Path
    tile_size: int
    stride: int
    hsv_s_thresh: int
    hsv_v_thresh: int
    background_max_ratio: float
    downscale_factor: float
    save_arrays: bool = False

    @property
    def processed_dir(self) -> Path:
        return self.output_root / self.name

def load_image_cv2(path):
    # Usa imdecode per evitare problemi con path Windows/Linux strani
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def load_mask_cv2(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

def process_slime_removal(img_bgr, mask_gray):
    """
    Rimuove l'inchiostro verde (marker) e corregge la maschera.
    Restituisce anche un flag che indica la presenza della macchia, così
    da poter scartare l'immagine se richiesto.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask_slime, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_solid = np.zeros_like(mask_slime)
    cv2.drawContours(mask_solid, contours, -1, (255), thickness=cv2.FILLED)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_solid_final = cv2.dilate(mask_solid, kernel, iterations=1)
    has_slime = np.count_nonzero(mask_solid_final) > 0

    # Inpainting (ricostruisce il tessuto sotto il verde)
    img_clean = cv2.inpaint(img_bgr, mask_solid_final, 3, cv2.INPAINT_TELEA)
    # Rimuove tumore dalla maschera se era coperto da inchiostro (per sicurezza)
    mask_clean = mask_gray.copy()
    mask_clean[mask_solid_final == 255] = 0
    return img_clean, mask_clean, has_slime

def is_patch_valid_hsv(img_crop, s_thresh, v_thresh, max_bg_ratio):
    """Controlla se c'è abbastanza tessuto (non solo vetro bianco)."""
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[:,:,1], hsv[:,:,2]
    # Background = Bassa Saturazione E Alta Luminosità
    mask_background = (s < s_thresh) & (v > v_thresh)
    ratio = np.count_nonzero(mask_background) / (img_crop.shape[0] * img_crop.shape[1])
    return ratio <= max_bg_ratio

def analyze_image_memory(img_bgr):
    """Controlla se l'immagine è corrotta (Shrek/Artefatti)."""
    if img_bgr is None: return "SAFE"
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Pixel non bianchi
    mask_fg = (hsv[:,:,1] > 40) & (hsv[:,:,2] < 250)
    if np.count_nonzero(mask_fg) < 100: return "SAFE"
    
    h_fg = hsv[:,:,0][mask_fg]
    # Range colori sospetti (marroni/verdi strani)
    shrek_ratio = np.count_nonzero((h_fg >= 10) & (h_fg < 80)) / np.count_nonzero(mask_fg)
    
    if shrek_ratio > 0.4: return "SHREK"
    return "SAFE"

def add_to_array(image, mask, array):
    """Salva in lista per creare il .npy finale."""
    image = image[..., ::-1] # BGR -> RGB
    mask = mask[..., np.newaxis]
    img4d = np.dstack((image, mask))
    array.append(img4d)

# =============================================================================
# ENGINE DI PROCESSING
# =============================================================================

def process_single_slide(img_path, mask_path, label, out_dirs, cfg, is_test=False):
    img_bgr = load_image_cv2(img_path)
    mask_gray = load_mask_cv2(mask_path)
    if img_bgr is None or mask_gray is None: return None

    # 1. Rimuovi Inchiostro (se presente) e segnala eventuali macchie
    img_clean, mask_clean, has_slime = process_slime_removal(img_bgr, mask_gray)
    if has_slime and not is_test:
        # Scarta completamente le immagini con macchia (sono duplicati senza macchia)
        cv2.imwrite(str(out_dirs['discard'] / img_path.name), img_clean)
        return "MACCHIA"
    
    # 2. Controllo Qualità (salta su test, dove non ci sono immagini sospette)
    if (not is_test) and analyze_image_memory(img_clean) == "SHREK":
        cv2.imwrite(str(out_dirs['discard'] / img_path.name), img_clean)
        return "SHREK"

    # 3. Downscale (Se richiesto dalla variante)
    if cfg.downscale_factor != 1.0:
        img_clean = cv2.resize(img_clean, None, fx=cfg.downscale_factor, fy=cfg.downscale_factor, interpolation=cv2.INTER_AREA)
        mask_clean = cv2.resize(mask_clean, None, fx=cfg.downscale_factor, fy=cfg.downscale_factor, interpolation=cv2.INTER_NEAREST)

    # 4. Tiling (Taglio in quadratini)
    h, w = img_clean.shape[:2]
    tiles_data = []
    img_array = []
    
    for y in range(0, h, cfg.stride):
        for x in range(0, w, cfg.stride):
            y_e, x_e = min(y + cfg.tile_size, h), min(x + cfg.tile_size, w)
            
            # Salta bordi troppo piccoli (minori di metà tile)
            if (y_e - y) < cfg.tile_size // 2 or (x_e - x) < cfg.tile_size // 2: continue
            
            crop_img = img_clean[y:y_e, x:x_e]
            crop_msk = mask_clean[y:y_e, x:x_e]
            
            # Padding (riempie i bordi se l'immagine finisce)
            pad_h, pad_w = cfg.tile_size - crop_img.shape[0], cfg.tile_size - crop_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop_img = cv2.copyMakeBorder(crop_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                crop_msk = cv2.copyMakeBorder(crop_msk, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            # Salva solo se c'è tessuto
            if is_patch_valid_hsv(crop_img, cfg.hsv_s_thresh, cfg.hsv_v_thresh, cfg.background_max_ratio):
                t_name = f"{img_path.stem}_y{y}_x{x}.png"
                
                cv2.imwrite(str(out_dirs['img'] / t_name), crop_img)
                cv2.imwrite(str(out_dirs['msk'] / t_name), crop_msk)
                
                if cfg.save_arrays:
                    add_to_array(crop_img, crop_msk, img_array)
                
                row = {
                    'sample_index': t_name,
                    'original_sample': img_path.name,
                    'tumor_coverage': cv2.countNonZero(crop_msk) / (cfg.tile_size**2)
                }
                if not is_test: row['label'] = label
                tiles_data.append(row)

    if cfg.save_arrays and img_array:
        np.save(out_dirs['arr'] / f"{img_path.stem}.npy", np.array(img_array))
        
    return tiles_data

def run_variant(variant_name, args):
    params = PREPROCESSING_VARIANTS[variant_name]
    save_arrays = args.save_arrays or params.get("save_arrays", False)
    cfg = PreprocessingConfig(
        name=variant_name,
        base_data=args.raw_data,
        output_root=args.output_dir,
        tile_size=params["tile_size"],
        stride=params["stride"],
        hsv_s_thresh=HSV_S_THRESH,
        hsv_v_thresh=HSV_V_THRESH,
        background_max_ratio=BACKGROUND_MAX_RATIO,
        downscale_factor=params.get("downscale_factor", 1.0),
        save_arrays=save_arrays,
    )

    # Setup cartelle output
    p_dir = cfg.processed_dir
    if p_dir.exists(): shutil.rmtree(p_dir)
    
    out_dirs = {
        'train_img': p_dir / "train/images",
        'train_msk': p_dir / "train/masks",
        'test_img': p_dir / "test/images",
        'test_msk': p_dir / "test/masks",
        'discard': p_dir / "discarded_shrek",
        'arr': p_dir / "arrays"
    }
    for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> 🚀 AVVIO VARIANTE: {variant_name}")
    print(f"    Descrizione: {params['description']}")
    print(f"    Salvataggio .npy: {'ON' if cfg.save_arrays else 'OFF'}")
    
    # --- TRAIN LOOP ---
    labels_path = cfg.base_data / "train_labels.csv"
    train_csv = []
    
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        # Loop su tutte le righe del CSV
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Training Slides"):
            fname = row['sample_index']
            
            # Cerca il file in sottocartelle
            found = list((cfg.base_data / "train_data").glob(f"**/{fname}"))
            if not found: continue
            img_p = found[0]
            
            # Cerca la maschera corrispondente
            mask_name = fname.replace("img_", "mask_").replace(".jpg", ".png")
            mask_p = img_p.parent / mask_name
            if not mask_p.exists(): continue

            res = process_single_slide(
                img_p, mask_p, row['label'], 
                {'img': out_dirs['train_img'], 'msk': out_dirs['train_msk'], 'discard': out_dirs['discard'], 'arr': out_dirs['arr']},
                cfg, is_test=False
            )
            if isinstance(res, list): train_csv.extend(res)
            
        pd.DataFrame(train_csv).to_csv(p_dir / "train_patches.csv", index=False)
        print(f"✅ Train completato: {len(train_csv)} tiles generati.")
    else:
        print("❌ train_labels.csv non trovato!")

    # --- TEST LOOP ---
    if not args.skip_test:
        test_csv = []
        test_files = sorted(list((cfg.base_data / "test_data").glob("**/img_*.*")))
        for img_p in tqdm(test_files, desc="Test Slides"):
            if "mask" in img_p.name: continue # Salta le maschere se trovate nel glob
            
            # Ricostruisci nome maschera test
            mask_name = f"mask_{img_p.stem.replace('img_', '')}.png"
            mask_p = img_p.parent / mask_name
            if not mask_p.exists(): continue

            res = process_single_slide(
                img_p, mask_p, None,
                {'img': out_dirs['test_img'], 'msk': out_dirs['test_msk'], 'discard': out_dirs['discard'], 'arr': out_dirs['arr']},
                cfg, is_test=True
            )
            if isinstance(res, list): test_csv.extend(res)
            
        pd.DataFrame(test_csv).to_csv(p_dir / "test_patches.csv", index=False)
        print(f"✅ Test completato: {len(test_csv)} tiles generati.")

    if args.zip_output:
        print(f"📦 Creazione ZIP per {variant_name}...")
        args.zip_dir.mkdir(parents=True, exist_ok=True)
        zip_name = args.zip_dir / variant_name
        shutil.make_archive(str(zip_name), 'zip', root_dir=args.output_dir, base_dir=variant_name)
        print(f"🎉 COMPLETATO: {zip_name}.zip pronto per l'upload!")
    else:
        print(f"ℹ️ ZIP saltato per {variant_name}; trovi i dati in {p_dir}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Script Locale di Preprocessing per Istologia")
    
    # Argomenti principali
    parser.add_argument("--variant", choices=list(PREPROCESSING_VARIANTS.keys()) + ["all"], default="all",
                        help="Scegli quale variante generare (o 'all' per tutte).")
    
    parser.add_argument("--raw-data", type=Path, default=DEFAULT_RAW_DATA, 
                        help="Cartella dove hai scaricato il dataset (deve contenere train_data, test_data, csv).")
    
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, 
                        help="Cartella temporanea di lavoro.")
    
    parser.add_argument("--zip-dir", type=Path, default=DEFAULT_ZIP_DIR, 
                        help="Cartella dove verranno salvati gli ZIP finali.")

    parser.add_argument("--zip-output", action="store_true", 
                        help="Se attivato crea lo zip finale e pulisce.")

    parser.add_argument("--save-arrays", action="store_true",
                        help="Se presente salva anche i .npy (di default disabilitato).")
    
    parser.add_argument("--skip-test", action="store_true", 
                        help="Se attivato, salta il processing del test set.")
    
    args = parser.parse_args()

    # Verifica preliminare
    if not args.raw_data.exists():
        print(f"\n❌ ERRORE: La cartella dati '{args.raw_data}' non esiste.")
        print(f"   Crea la cartella '{args.raw_data}' e mettici dentro 'train_data', 'test_data' e i csv.")
        return

    # Selezione varianti
    variants_to_run = list(PREPROCESSING_VARIANTS.keys()) if args.variant == "all" else [args.variant]
    
    print(f"=== INIZIO PREPROCESSING LOCALE ===")
    print(f"Dataset Sorgente: {args.raw_data}")
    print(f"Output ZIP:       {args.zip_dir}")
    print(f"Varianti:         {variants_to_run}")
    
    for v in variants_to_run:
        try:
            run_variant(v, args)
        except Exception as e:
            print(f"❌ ERRORE CRITICO sulla variante {v}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
