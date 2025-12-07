from joblib import Parallel, delayed
import os

# ... [Keep all your imports and configuration constants] ...
# ... [Keep load_image_cv2, load_mask_cv2, etc.] ...
# ... [Keep process_slime_removal, analyze_image_memory, is_patch_valid_hsv] ...

# ... [Keep your FIXED process_single_slide function] ...

# =============================================================================
# HELPER: WRAPPER FOR PARALLEL EXECUTION
# =============================================================================
def process_wrapper(row_data, output_dirs):
    """
    A wrapper to unpack arguments for joblib.
    """
    # Unpack arguments
    img_path, mask_path, label, is_test = row_data
    out_img, out_mask, discard, arrays = output_dirs
    
    # Important: Prevent OpenCV from spawning its own threads inside a process
    # This prevents CPU thrashing.
    cv2.setNumThreads(0)
    
    return process_single_slide(
        img_path, mask_path, label, 
        out_img, out_mask, discard, arrays, 
        is_test_set=is_test
    )

# =============================================================================
# 5. MAIN EXECUTION (PARALLELIZED)
# =============================================================================

def main():
    base_data = Path("../../drive/MyDrive/AN2DL_Challenge2-TheBigBatchTheory/data/dataset")
    train_dir = base_data / "train_data"
    labels_csv = base_data / "train_labels.csv"

    # Output Directories
    processed_dir = base_data / "testpreprocessing"
    
    out_train_img = processed_dir / "train/images"
    out_train_mask = processed_dir / "train/masks"
    out_test_img = processed_dir / "test/images"
    out_test_mask = processed_dir / "test/masks"
    discard_dir = processed_dir / "discarded_shrek"
    arrays_dir = processed_dir / "arrays"

    # Create directories
    for d in [out_train_img, out_train_mask, out_test_img, out_test_mask, discard_dir, arrays_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FASE 1: TRAINING SET PREPARATION
    # -------------------------------------------------------------------------
    print(">>> FASE 1: Preparing TRAINING Tasks...")
    
    tasks = [] # List to hold all work to be done
    
    if labels_csv.exists() and train_dir.exists():
        labels_df = pd.read_csv(labels_csv).sort_values(by='sample_index')
        
        # 1. PRE-CALCULATE PATHS (Fast, Single-threaded)
        # We assume the file finding is fast enough to do sequentially.
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Locating Files"):
            fname = row['sample_index']
            label = row['label']
            
            # Locate image
            img_path = train_dir / fname
            if not img_path.exists(): 
                 found = list(train_dir.glob(f"**/{fname}")) 
                 if found: img_path = found[0]
                 else: continue
            
            # Locate mask
            mask_name = fname.replace("img_", "mask_") 
            mask_path = img_path.parent / mask_name
            if not mask_path.exists(): 
                 mask_path = img_path.parent / fname.replace("img_", "mask_").replace(".jpg", ".png")
                 if not mask_path.exists(): continue

            # Add to task list
            # Format: (img_path, mask_path, label, is_test_set)
            tasks.append((img_path, mask_path, label, False))
            
    else:
        print("⚠️ Training data not found.")
        return

    # -------------------------------------------------------------------------
    # EXECUTE PARALLEL PROCESSING
    # -------------------------------------------------------------------------
    print(f"\n>>> Starting Multiprocessing on {len(tasks)} slides...")
    print(f"CPU Cores available: {os.cpu_count()}")
    
    # Bundle output directories
    out_dirs = (out_train_img, out_train_mask, discard_dir, arrays_dir)

    # RUN PARALLEL
    # n_jobs=-1 uses all available cores. 
    # n_jobs=-2 uses all except one (good if you want to keep the PC usable).
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_wrapper)(task, out_dirs) 
        for task in tqdm(tasks, desc="Processing Tiles")
    )

    # -------------------------------------------------------------------------
    # AGGREGATE RESULTS
    # -------------------------------------------------------------------------
    train_rows = []
    
    for res in results:
        if res is None: continue # Skip empty/invalid slides
        if res == "SHREK": continue # Skip discarded slides
        if isinstance(res, list):
            train_rows.extend(res)

    # Save CSV
    if train_rows:
        train_df = pd.DataFrame(train_rows)
        cols = ['sample_index', 'original_sample', 'label', 'tumor_coverage']
        train_df = train_df[cols]
        train_df.to_csv(processed_dir / "train_patches.csv", index=False)
        print(f"✅ Training Tiles Saved: {len(train_rows)}")
    else:
        print("⚠️ No tiles generated.")

if __name__ == "__main__":
    main()