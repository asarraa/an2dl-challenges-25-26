import cv2
import numpy as np
import argparse
import sys
import shutil
import pandas as pd
from pathlib import Path

# --- 1. FUNZIONI DI UTILITY & IO ---

def load_image_cv2(path):
    # Usa imdecode per gestire path con caratteri speciali/OS diversi
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def load_mask_cv2(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return img

# --- 2. LOGICA RIMOZIONE SLIME (Preprocessing) ---

def process_slime_removal(img_bgr, mask_gray):
    """
    Rimuove le macchie verdi dall'immagine (inpainting)
    e aggiorna la maschera corrispondente (mette a 0 i pixel rimossi).
    """
    # Conversione per rilevamento
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Range Verde (Slime)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)

    # Pulizia maschera slime
    contours, _ = cv2.findContours(mask_slime, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_solid = np.zeros_like(mask_slime)
    cv2.drawContours(mask_solid, contours, -1, (255), thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_solid_final = cv2.dilate(mask_solid, kernel, iterations=1)

    # Inpainting sull'immagine originale
    img_clean = cv2.inpaint(img_bgr, mask_solid_final, 3, cv2.INPAINT_TELEA)

    # Aggiornamento della maschera di segmentazione (rimuove l'area dello slime)
    mask_clean = mask_gray.copy()
    mask_clean[mask_solid_final == 255] = 0
    
    return img_clean, mask_clean

# --- 3. LOGICA CLASSIFICATORE V11 (Shrek Dominance) ---

def analyze_image_memory(img_bgr):
    """
    Versione adattata di analyze_image_v11 che accetta 
    un array numpy (immagine già in memoria) invece di un path.
    """
    if img_bgr is None: return None
    
    # Converti in RGB per coerenza con la logica originale, poi HSV
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # Nota: hsv da BGR standard
    
    # --- 1. RILEVAMENTO FOREGROUND ---
    mask_foreground = (hsv[:,:,1] > 40) & (hsv[:,:,2] < 250)
    foreground_pixels = np.count_nonzero(mask_foreground)
    
    if foreground_pixels < 100:
        return "SAFE (Empty)", 0.0, 0.0, 0.0

    h_foreground = hsv[:,:,0][mask_foreground]

    # --- 2. CONTEGGIO COLORI ---
    # A. TESSUTO (Rosa/Viola): H 125-175
    count_tissue = np.count_nonzero((h_foreground >= 125) & (h_foreground <= 175))
    
    # B. INCHIOSTRO (Verde Freddo): H 80-120
    count_ink = np.count_nonzero((h_foreground >= 80) & (h_foreground < 125))
    
    # C. SHREK (Skin + Clothes)
    count_shrek_skin = np.count_nonzero((h_foreground >= 20) & (h_foreground < 80))
    count_shrek_clothes = np.count_nonzero((h_foreground >= 10) & (h_foreground < 20))
    count_shrek_total = count_shrek_skin + count_shrek_clothes
    
    # --- 3. CALCOLO RAPPORTI ---
    ratio_tissue = count_tissue / foreground_pixels
    ratio_ink = count_ink / foreground_pixels
    ratio_shrek = count_shrek_total / foreground_pixels

    if count_tissue > 0:
        shrek_dominance = count_shrek_total / count_tissue
    else:
        shrek_dominance = 999.0

    # --- 4. LOGICA DI CLASSIFICAZIONE V11 ---

    # REGOLA 1: INK SALVATION
    if ratio_ink > ratio_shrek and ratio_ink > 0.1:
        return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

    # REGOLA 2: SHREK DOMINANCE
    if ratio_shrek > 0.4 and shrek_dominance > 4.0:
        return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance

    # REGOLA 3: TISSUE SAFETY
    if ratio_tissue > 0.05:
        return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

    # REGOLA 4: SHREK FALLBACK
    if ratio_shrek > 0.3:
        return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance

    return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

# --- 4. MAIN PIPELINE ---

def main():
    
    
    input_dir = Path("./data/train_data")
    output_dir = Path("./data/processed")
    labels_dir = Path("./data/train_labels.csv")

    labels = pd.read_csv(labels_dir)
    
    
    # Creazione struttura cartelle output
    final_img_dir = output_dir / "images"
    final_mask_dir = output_dir / "masks"
    discard_dir = output_dir / "discarded_shrek"

    final_img_dir.mkdir(parents=True, exist_ok=True)
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    discard_dir.mkdir(parents=True, exist_ok=True)

    # Trova tutte le immagini
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for ext in extensions:
        img_files.extend(list(input_dir.glob(f"**/{ext}")))

    # Filtra solo quelli che matchano il pattern img_XXXX
    img_files = [f for f in img_files if f.name.startswith("img_") and "mask" not in f.name]

    print(f"Trovate {len(img_files)} immagini da processare.")
    print("-" * 50)

    stats = {"SAFE": 0, "SHREK": 0, "ERROR": 0}
    report_rows = []

    for i, img_path in enumerate(img_files):
        try:
            # 1. Identifica Mask path
            # Assume formato: img_1234.png -> mask_1234.png
            file_stem = img_path.stem # img_1234
            suffix = img_path.suffix  # .png
            # Estrae il numero (o la parte dopo img_)
            id_part = file_stem.replace("img_", "")
            mask_name = f"mask_{id_part}{suffix}"
            mask_path = img_path.parent / mask_name

            if not mask_path.exists():
                # Prova fallback estensione png se originale era jpg
                mask_path = img_path.parent / f"mask_{id_part}.png"
                if not mask_path.exists():
                    print(f"⚠️ Mask non trovata per {img_path.name}, salto.")
                    stats["ERROR"] += 1
                    continue

            # 2. Carica Immagine e Maschera
            img_bgr = load_image_cv2(img_path)
            mask_gray = load_mask_cv2(mask_path)

            if img_bgr is None or mask_gray is None:
                print(f"Errore caricamento file per {img_path.name}")
                stats["ERROR"] += 1
                continue

            # 3. RIMOZIONE SLIME
            img_clean, mask_clean = process_slime_removal(img_bgr, mask_gray)
            
            # 4. CLASSIFICAZIONE V11 (Sull'immagine pulita!)
            cls, r_tiss, r_shrek, dom = analyze_image_memory(img_clean)

            # Logging
            report_rows.append({
                "filename": img_path.name,
                "classification": cls,
                "pink_ratio": r_tiss,
                "shrek_ratio": r_shrek,
                "dominance": dom
            })

            # 5. SALVATAGGIO
            if cls == "SHREK":
                # Sposta/Salva nei scarti
                cv2.imwrite(str(discard_dir / img_path.name), img_clean)
                cv2.imwrite(str(discard_dir / mask_name), mask_clean)
                stats["SHREK"] += 1
                #i want to remove the corresponding row from labels dataframe
                labels.drop(labels[labels['sample_index'] == img_path.name].index, inplace=True)
                print(f"❌ {img_path.name} -> SHREK (Scartato)")
            else:
                # SAFE -> Salva nelle cartelle finali divise
                cv2.imwrite(str(final_img_dir / img_path.name), img_clean)
                cv2.imwrite(str(final_mask_dir / mask_name), mask_clean)
                stats["SAFE"] += 1
                # print(f"✅ {img_path.name} -> SAFE") # Decommenta per log verbose

        except Exception as e:
            print(f"Errore critico su {img_path.name}: {e}")
            stats["ERROR"] += 1
        
        # Avanzamento
        if i % 20 == 0:
            print(f"Processati {i}/{len(img_files)}...", end="\r")

    
    # Salva labels in un file csv in /processed
    labels.to_csv(output_dir / "train_labels_processed.csv", index=False)
    
    print("\n" + "="*50)
    print("ELABORAZIONE COMPLETATA")
    print(f"📁 Output: {output_dir}")
    print(f"✅ Immagini SAFE (Salvate): {stats['SAFE']}")
    print(f"❌ Immagini SHREK (Scartate): {stats['SHREK']}")
    print(f"⚠️ Errori (No Mask/Corrotte): {stats['ERROR']}")
    print("="*50)

if __name__ == "__main__":
    main()