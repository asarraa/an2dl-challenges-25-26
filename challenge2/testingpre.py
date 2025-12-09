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

# Size of the square patch extracted from the slide.
# 512x512 is a good trade-off between context and resolution.
TILE_SIZE = 224

# Stride determines the overlap between patches.
# A stride of 256 (half of TILE_SIZE) means 50% overlap.
# This acts as Test-Time Augmentation (TTA) ensuring every cell is centered at least once.
STRIDE = 112

# HSV Thresholds for detecting "Glass" (Background).    
# Tissue usually has higher saturation. Background is white/grey (Low Saturation, High Value).
HSV_S_THRESH = 15   
HSV_V_THRESH = 200  

# Maximum allowed ratio of background pixels in a patch.
# If more than 50% of the patch is background (glass), it is discarded.
BACKGROUND_MAX_RATIO = 0.5 


# =============================================================================
# 1. UTILITY & I/O FUNCTIONS
# =============================================================================

def load_image_cv2(path):
    """
    Loads an image from a file path using OpenCV.
    
    Args:
        path (Path or str): Path to the image file.
        
    Returns:
        np.array: Loaded image in BGR format, or None if loading fails.
    """
    # cv2.imdecode is used instead of cv2.imread to correctly handle 
    # file paths with special characters or different OS encodings.
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def load_mask_cv2(path):
    """
    Loads a segmentation mask (grayscale).
    
    Args:
        path (Path or str): Path to the mask file.
        
    Returns:
        np.array: Loaded mask in Grayscale format.
    """
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# =============================================================================
# 2. PREPROCESSING LOGIC (SLIME REMOVAL & TISSUE VALIDATION)
# =============================================================================
def contains_slime(img_bgr, threshold=50):
    # Convert BGR to HSV color space to easily isolate the green color.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the green color (Marker Ink).
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create a binary mask where green pixels are white (255).
    mask_slime = cv2.inRange(hsv, lower_green, upper_green)
    
    # Count the number of green pixels detected
    green_pixel_count = cv2.countNonZero(mask_slime)
    
    # Determine if the count exceeds the threshold
    return green_pixel_count > threshold

def is_patch_valid_hsv(img_crop):
    """
    Determines if a patch contains enough valid biological tissue.
    
    Why: We want to avoid training/inference on empty white glass.
    Criterion: Calculates the ratio of background pixels based on HSV thresholds.
    
    Args:
        img_crop (np.array): The 512x512 image patch.
        
    Returns:
        bool: True if the patch is valid (contains tissue), False otherwise.
    """
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1] # Saturation channel
    v = hsv[:,:,2] # Value (Brightness) channel
    
    # Definition of "Background" (Glass/White space):
    # Low Saturation (grey/white) AND High Brightness.
    mask_background = (s < HSV_S_THRESH) & (v > HSV_V_THRESH)
    
    # Calculate ratio of background pixels relative to total area.
    background_ratio = np.count_nonzero(mask_background) / (img_crop.shape[0] * img_crop.shape[1])
    
    # Keep the patch only if the background is less than the threshold (e.g., 50%).
    return background_ratio <= BACKGROUND_MAX_RATIO

# =============================================================================
# 3. QUALITY CONTROL (SHREK / ARTIFACT DETECTION)
# =============================================================================

def analyze_image_memory(img_bgr):
    """
    Analyzes the image to detect if it's corrupted by excessive artifacts ("Shrek").
    Implements the "V11" logic based on color ratios.
    
    Args:
        img_bgr (np.array): The cleaned image.
        
    Returns:
        tuple: (Classification string, pink_ratio, shrek_ratio, dominance_score)
    """
    if img_bgr is None: return "SAFE", 0, 0, 0
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 
    
    # 1. Identify Foreground (everything that is not white background)
    mask_foreground = (hsv[:,:,1] > 15) & (hsv[:,:,2] < 250)
    foreground_pixels = np.count_nonzero(mask_foreground)
    
    if foreground_pixels < 100:
        return "SAFE (Empty)", 0.0, 0.0, 0.0

    # Extract Hue values for foreground pixels
    h_foreground = hsv[:,:,0][mask_foreground]

    # 2. Count pixels for specific materials based on Hue ranges:
    # Tissue (Pink/Purple): Hue 125-175
    count_tissue = np.count_nonzero((h_foreground >= 125) & (h_foreground <= 175))
    
    # Green Ink (Cold Green): Hue 80-120
    count_ink = np.count_nonzero((h_foreground >= 80) & (h_foreground < 125))
    
    # "Shrek" (Artifacts, Skin, Clothes): Low Hues 10-80
    count_shrek_skin = np.count_nonzero((h_foreground >= 20) & (h_foreground < 80))
    count_shrek_clothes = np.count_nonzero((h_foreground >= 10) & (h_foreground < 20))
    count_shrek_total = count_shrek_skin + count_shrek_clothes
    
    # 3. Calculate Ratios relative to foreground
    ratio_tissue = count_tissue / foreground_pixels
    ratio_ink = count_ink / foreground_pixels
    ratio_shrek = count_shrek_total / foreground_pixels

    if count_tissue > 0:
        shrek_dominance = count_shrek_total / count_tissue
    else:
        shrek_dominance = 999.0

    # 4. Classification Rules (V11 Logic)
    
    # Rule: If Ink is dominant but we saved it via cleaning, it might be safe.
    if ratio_ink > ratio_shrek and ratio_ink > 0.1:
        return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

    # Rule: If "Shrek" artifacts are overwhelming compared to tissue.
    if ratio_shrek > 0.4 and shrek_dominance > 4.0:
        return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance

    # Rule: If there is a decent amount of tissue, assume safe.
    if ratio_tissue > 0.05:
        return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

    # Fallback: If Shrek ratio is moderately high.
    if ratio_shrek > 0.3:
        return "SHREK", ratio_tissue, ratio_shrek, shrek_dominance

    return "SAFE", ratio_tissue, ratio_shrek, shrek_dominance

# =============================================================================
# 4. TILING ENGINE
# =============================================================================

def process_single_slide(img_path, mask_path, label, output_img_dir, output_mask_dir, is_test_set=False):
    """
    Orchestrates the processing pipeline for a single whole-slide image (WSI) or ROI.
    Pipeline: Load -> Remove Slime -> Check Quality (Shrek) -> Tile -> Save.
    
    Args:
        img_path (Path): Path to source image.
        mask_path (Path): Path to source mask.
        label (str): Class label (e.g., "Luminal A"). None for Test Set.
        output_img_dir (Path): Directory to save patches.
        output_mask_dir (Path): Directory to save mask patches.
        discard_dir (Path): Directory to save rejected images (Shrek).
        is_test_set (bool): Flag to indicate if we are processing test data.
        
    Returns:
        list: A list of dictionaries containing metadata for the generated tiles.
              Returns "SHREK" string if the image was discarded.
    """
    img = load_image_cv2(img_path)
    mask = load_mask_cv2(mask_path)
    
    # Basic error check
    if img is None or mask is None: 
        return None

    if not is_test_set:
        # --- Step 1: Slime Removal ---
        if contains_slime(img):
            return
        #img_clean, mask_clean = process_slime_removal(img_bgr, mask_gray)

        # --- Step 2: Quality Control (Shrek Check) ---
        cls, _, _, _ = analyze_image_memory(img)
        
        # If classified as "SHREK" (corrupted/artifact), we discard the whole slide.
        if cls == "SHREK":
            # Save the full clean image to discard folder for manual inspection
            #cv2.imwrite(str(discard_dir / img_path.name), img)
            # Even for test set, we flag it. 
            return "SHREK"

    # --- Step 3: Tiling (Patch Extraction) ---
    tiles_data = []
    h, w, _ = img.shape
    base_name = img_path.stem # e.g., "img_001"

    # Iterate over the image with the defined stride
    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            
            # Skip strips at the edges that are too small (less than half a tile)
            if (y_end - y) < TILE_SIZE // 2 or (x_end - x) < TILE_SIZE // 2: continue

            # Extract the crop
            img_crop = img[y:y_end, x:x_end]
            mask_crop = mask[y:y_end, x:x_end]

            # --- Padding Logic ---
            # If the crop is smaller than TILE_SIZE (at the edges), we need to pad.
            pad_h = TILE_SIZE - img_crop.shape[0]
            pad_w = TILE_SIZE - img_crop.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                # REFLECTION PADDING for Image:
                # Mirrors the edge pixels. This maintains texture continuity and prevents
                # "hard edge" artifacts that zero-padding creates (CNNs hate hard edges).
                img_crop = cv2.copyMakeBorder(img_crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                
                # CONSTANT PADDING for Mask:
                # We pad the mask with 0 (Background) because we cannot "hallucinate" tumor presence.
                mask_crop = cv2.copyMakeBorder(mask_crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            # --- Step 4: Validity Check & Saving ---
            # Check if the patch contains tissue using HSV analysis.
            if is_patch_valid_hsv(img_crop):
                
                # Calculate 'tumor_coverage':
                # This is the weight used later for Majority Voting.
                # It represents the percentage of tumor pixels in this specific patch.
                tumor_coverage = cv2.countNonZero(mask_crop) / (TILE_SIZE * TILE_SIZE)
                
                # Construct unique filename for the tile
                tile_name = f"{base_name}_y{y}_x{x}.png"
                
                # Save to disk
                cv2.imwrite(str(output_img_dir / tile_name), img_crop)
                cv2.imwrite(str(output_mask_dir / tile_name), mask_crop)

                # Prepare metadata for CSV
                row = {
                    'sample_index': tile_name,      # name of the tile: image_name_x_y coorindates
                    'original_sample': img_path.name, # Name of the full image
                    'tumor_coverage': tumor_coverage  # Weight for inference
                }
                
                # Add label only if we are in Training mode
                if not is_test_set:
                    row['label'] = label 
                
                tiles_data.append(row)
            
    if len(tiles_data) == 0:
        print(f"No valid tiles extracted from {img_path.name} after processing.")
    return tiles_data

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main(do_test=True, preprocess_name=None):
    '''
    do_test: if applying also to
    '''
    if preprocess_name == None:
        print("[ERROR] Give a name to this preprocessing")
        return

    # Output Directories
    single_preprocessing_dir = config.BASE_PREPROCESSED / preprocess_name
    
    # Create specific subdirectories for organized output
    #Train
    out_train_img = single_preprocessing_dir / "train/images"
    out_train_mask = single_preprocessing_dir / "train/masks"
    
    #Test
    out_test_img = single_preprocessing_dir / "test/images"
    out_test_mask = single_preprocessing_dir / "test/masks"

    '''
    data --> BASE_DATA
    -dataset --> base_dataset
        --train_data
        --test_data
        --train_labels.csv
        
    -preprocessed --> base_preprocessed 
        --<Preprocess_name> --> single_preprocessing_dir
            ---train
                ---images
                ---mask
                ---arrays
                ---.csv
            ---test
                ---images
                ---mask
                ---arrays
                ---.csv
            ---discared
                ---images
                ---mask
    '''
    # Clean up previous runs and create directories
    for d in [out_train_img, out_train_mask]:
        if d.exists(): 
            shutil.rmtree(d)
            print(f"[DEBUG] testing_pre: Cleaning output directory {d}!!!!!")
        d.mkdir(parents=True, exist_ok=True)

    if do_test:
        for d in [out_test_img, out_test_mask]:
            if d.exists(): 
                shutil.rmtree(d)
                print(f"[DEBUG] testing_pre: Cleaning output directory {d}!!!!!")
            d.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------------------------------
    # FASE 1: TRAINING SET PROCESSING
    # -------------------------------------------------------------------------
    print(">>> FASE 1: Processing TRAINING SET (Labels + Weights)")
    labels_csv = config.LABELS_CSV
    train_dir = config.TRAIN_DIR
    if labels_csv.exists() and train_dir.exists():
        labels_df = pd.read_csv(labels_csv)
        
        # SORTING: Sort the dataframe by sample_index to ensure processing order is ascending.
        # This guarantees that img_001 is processed before img_002.
        labels_df = labels_df.sort_values(by='sample_index')
        
        train_rows = []
        
        # Iterate through the sorted labels
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Training Slides"):
            fname = row['sample_index']
            label = row['label']
            
            # Locate the image file (recursive search allows subfolders)
            img_path = train_dir / fname
            if not img_path.exists(): 
                 found = list(train_dir.glob(f"**/{fname}")) 
                 if found: img_path = found[0]
                 else: continue
            
            # Locate the corresponding mask
            # Assumes naming convention: img_X -> mask_X
            mask_name = fname.replace("img_", "mask_") 
            mask_path = img_path.parent / mask_name
            
            # Fallback if mask is .png but image is .jpg
            if not mask_path.exists(): 
                 mask_path = img_path.parent / fname.replace("img_", "mask_").replace(".jpg", ".png")
                 if not mask_path.exists(): continue

            # Process the slide
            res = process_single_slide(
                img_path, mask_path, label, 
                out_train_img, out_train_mask,
                is_test_set=False,
            )
            
            # If successful (list returned), add rows to the dataset
            if isinstance(res, list): 
                train_rows.extend(res)

        # Save the final CSV for training
        if train_rows:
            train_df = pd.DataFrame(train_rows)
            # Reorder columns for clarity
            cols = ['sample_index', 'original_sample', 'label', 'tumor_coverage']
            train_df = train_df[cols]
            train_df.to_csv(single_preprocessing_dir / "train_patches.csv", index=False)
            print(f"✅ Training Tiles Saved: {len(train_rows)}")
        else:
            print("⚠️ No tiles generated for Training Set.")
    else:
        print("⚠️ train_data folder or train_labels.csv not found.")
        
    if do_test and config.TEST_DIR.exists():
        process_test(preprocess_name)

 #-------------------------------------------------------------------------
 #FASE 2: TEST SET PROCESSING
 #-------------------------------------------------------------------------

def process_test(preprocess_name=None):

    if preprocess_name == None :
        print("[ERROR] Give a name to this preprocessing")
        return

    #subfolder of preprocessed
    single_preprocessing_dir = config.BASE_PREPROCESSED / preprocess_name

    #output directories
    out_test_img = single_preprocessing_dir / "test/images"
    out_test_mask = single_preprocessing_dir / "test/masks"

    print("\n>>> FASE 2: Processing TEST SET (Weights Only - No Labels)")
    test_dir = config.TEST_DIR
    if test_dir.exists():
        test_rows = []
    
        # Find all images.
        # SORTING: We gather all files first, then sort them to ensure ascending order.
        all_files = sorted(list(test_dir.glob("**/img_*.*")))
    
        for img_path in tqdm(all_files, desc="Test Slides"):
            # Skip if glob accidentally picked up a mask file 
            if "mask" in img_path.name: continue #(img_path.name elimina ".png")
        
            # Find corresponding mask
            id_part = img_path.stem.replace("img_", "")
            mask_path = img_path.parent / f"mask_{id_part}{img_path.suffix}"
            if not mask_path.exists():
                mask_path = img_path.parent / f"mask_{id_part}.png"
        
            if not mask_path.exists(): continue #if mask not found, skip image
            # Process the slide (Test Mode: label=None, is_test_set=True)
            res = process_single_slide( #call process single slide with mask and corresponding mask
                img_path, mask_path, None, 
                out_test_img, out_test_mask, #directories
                is_test_set=True
            )
        
            if isinstance(res, list): 
                test_rows.extend(res)
        # Save the final CSV for testing
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            # Reorder columns (No Label column here)
            cols = ['sample_index', 'original_sample', 'tumor_coverage']
            test_df = test_df[cols]
            test_df.to_csv(single_preprocessing_dir / "test_patches.csv", index=False)
            print(f"✅ Test Tiles Saved: {len(test_rows)}")
        else:
            print("⚠️ No tiles generated for Test Set.")
    else:
        print("⚠️ test_data folder not found.")