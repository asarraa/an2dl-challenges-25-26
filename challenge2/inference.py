import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader 
import tqdm

# Assumiamo che questi file esistano e siano importabili
import mil_pipeline as data_loaders
import models
# =============================================================================
# --- FUNZIONE PRINCIPALE DI INFERENZA ---
# =============================================================================

def make_inference(
    model_path: Path,
    base_path: Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
    experiment_id: str,
    inference_type: str = "weighted_majority_vote"
):
    """
    Esegue l'intero pipeline di inferenza: caricamento dati, caricamento modello,
    predizione per tile e aggregazione per slide.
    """
    print(f"\n[INFO] Starting inference for experiment '{experiment_id}'...")
    print(f"[INFO] Using aggregation mode: '{inference_type}'")
    
    # 1. Carica i dati del test set
    try:
        test_loader = data_loaders.get_mil_loaders(
            base_path=base_path,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to create test data loader. {e}")
        return

    # 2. Carica il modello addestrato
    # --- MODIFICA 1: Semplificata la logica di caricamento del modello ---
    try:
        # Prima definiamo la mappa delle label per la submission finale
        train_csv_path = base_path / "train" / "train_patches.csv"
        if not train_csv_path.exists():
            raise FileNotFoundError("train_patches.csv not found, cannot create inverse label map.")
        
        df_train = pd.read_csv(train_csv_path)
        unique_labels = sorted(df_train['label'].unique())
        label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
        inv_label_map = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)

        print(f"[INFO] Loading model '{model_name}' with {num_classes} classes.")
        if model_name in ("MultiScale", "DualBranchResNet"):
            model = models.DualBranchResNet(num_classes=num_classes)
        else:
            # Aggiungi qui la logica per altri modelli se necessario
            raise ValueError(f"Unsupported model name '{model_name}' for this script.")

        print(f"[INFO] Loading weights from: '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("[INFO] Model loaded successfully.")

    except Exception as e:
        print(f"❌ ERROR: Failed to load model. {e}")
        return

    # 3. Esegui le predizioni per ogni tile
    tile_preds_df = predict_tiles(model, test_loader, device, num_classes)

    # 4. Aggrega i risultati a livello di slide
    meta_df = pd.read_csv(base_path / "test" / "test_patches.csv")

    if inference_type == "weighted_majority_vote":
        submission_df = weighted_mean_aggregation(tile_preds_df, meta_df, inv_label_map)
    elif inference_type == "mil":
        submission_df = mil_aggregation(tile_preds_df, meta_df, inv_label_map)
    else:
        raise ValueError(f"Inference type '{inference_type}' non supportato.")

    # 5. Salva la submission
    output_path = Path(f"./{experiment_id}_submission.csv") # Salva nella directory corrente
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Inference complete. Submission file saved to '{output_path}'")


# =============================================================================
# --- HELPERS DI INFERENZA ---
# =============================================================================

def predict_tiles(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> pd.DataFrame:
    """
    Esegue l'inferenza per ogni tile nel DataLoader e restituisce un DataFrame con le probabilità.
    """
    model.eval()
    all_names = []
    all_probs = []

    with torch.no_grad():
        for context_batch, detail_batch, names_batch in tqdm.tqdm(loader, desc="Predicting on tiles"):
            # --- MODIFICA 2: Gestione esplicita dell'input multi-scala ---
            context_batch = context_batch.to(device)
            detail_batch = detail_batch.to(device)
            
            # Passa la tupla di tensori al modello
            logits = model((context_batch, detail_batch))
            probs = F.softmax(logits, dim=1)
            
            all_names.extend(names_batch)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    
    # Crea il DataFrame con i risultati
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    data = {"sample_index": all_names}
    for i, col in enumerate(prob_cols):
        data[col] = all_probs[:, i]

    return pd.DataFrame(data)


def weighted_mean_aggregation(tile_df: pd.DataFrame, meta_df: pd.DataFrame, inv_label_map: Dict[int, str]) -> pd.DataFrame:
    """Aggregazione tramite media delle probabilità."""
    # --- MODIFICA 3: Semplificata la logica di merge e aggregazione ---
    merged = pd.merge(tile_df, meta_df[['sample_index', 'original_sample']], on="sample_index")
    
    prob_cols = [col for col in tile_df.columns if col.startswith('prob_')]
    
    # Raggruppa per slide originale e calcola la media delle probabilità
    slide_probs = merged.groupby("original_sample")[prob_cols].mean()
    
    # Trova la classe con la probabilità media più alta per ogni slide
    best_class_idx = slide_probs.idxmax(axis=1).str.replace('prob_', '').astype(int)
    
    submission_df = pd.DataFrame({
        "sample_index": best_class_idx.index,
        "label": best_class_idx.map(inv_label_map)
    }).sort_values("sample_index")
    
    return submission_df


def mil_aggregation(tile_df: pd.DataFrame, meta_df: pd.DataFrame, inv_label_map: Dict[int, str]) -> pd.DataFrame:
    """Aggregazione tramite Max Pooling (Multiple Instance Learning)."""
    merged = pd.merge(tile_df, meta_df[['sample_index', 'original_sample']], on="sample_index")
    prob_cols = [col for col in tile_df.columns if col.startswith('prob_')]

    # Raggruppa per slide e trova la probabilità MASSIMA per ogni classe
    slide_max_probs = merged.groupby("original_sample")[prob_cols].max()

    # Trova la classe con la probabilità massima più alta
    best_class_idx = slide_max_probs.idxmax(axis=1).str.replace('prob_', '').astype(int)

    submission_df = pd.DataFrame({
        "sample_index": best_class_idx.index,
        "label": best_class_idx.map(inv_label_map)
    }).sort_values("sample_index")
    
    return submission_df

# =============================================================================
# --- Esempio di come chiamare la funzione nel tuo notebook ---
# =============================================================================
#if __name__ == '__main__':
#     # Questo blocco serve solo per un esempio di esecuzione
#     
#     # 1. Definisci i parametri
#     MODEL_PATH = Path("./fit_models/NOME_DEL_TUO_ESPERIMENTO_best_model.pth")
#     BASE_DATA_PATH = Path("/path/to/your/multiscale_preprocessed_data")
#     MODEL_NAME = "MultiScale" # o "DualBranchResNet"
#     BATCH_SIZE = 32
#     EXPERIMENT_ID = "multiscale_inference_01"
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. Chiama la funzione principale di inferenza
#     make_inference(
#         model_path=MODEL_PATH,
#         base_path=BASE_DATA_PATH,
#         model_name=MODEL_NAME,
#         device=DEVICE,
#         batch_size=BATCH_SIZE,
#         experiment_id=EXPERIMENT_ID,
#         inference_type="weighted_majority_vote" # o "mil"
#     )