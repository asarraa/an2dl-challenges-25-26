import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader 
from tqdm import tqdm
from typing import List, Dict


# Assumiamo che questi file esistano e siano importabili nel tuo ambiente
import mil_pipeline as data_loaders
import mil_model as models

# =============================================================================
# --- FUNZIONE PRINCIPALE DI INFERENZA (PER MODELLO MIL) ---
# =============================================================================

def make_mil_inference(
    model_path: Path,
    base_path: Path,
    device: torch.device,
    batch_size: int,
    experiment_id: str,
    backbone_name: str = 'vit_small_patch16_224' # Aggiunto per flessibilità
):
    """
    Esegue l'inferenza con un modello MIL.
    """
    print(f"\n[INFO] Starting MIL inference for experiment '{experiment_id}'...")
    
    # 1. Carica il test loader per il MIL
    try:
        test_loader = data_loaders.get_mil_test_loader(
            base_path=base_path,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to create MIL test data loader. {e}")
        return

    # 2. Carica il modello MIL addestrato
    try:
        # Ricostruisci la mappa delle label per la submission finale
        train_csv_path = base_path / "train" / "train_patches.csv"
        df_train = pd.read_csv(train_csv_path)
        unique_labels = sorted(df_train['label'].unique())
        label_map = {label_str: i for i, label_str in enumerate(unique_labels)}
        inv_label_map = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)

        print(f"[INFO] Loading AttentionMIL model with {num_classes} classes and '{backbone_name}' backbone.")
        
        # Istanzia il modello corretto
        model = models.AttentionMIL(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=False # Non serve scaricare i pesi, li carichiamo dal file
        )

        print(f"[INFO] Loading weights from: '{model_path}'")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("[INFO] Model loaded successfully.")

    except Exception as e:
        print(f"❌ ERROR: Failed to load model. {e}")
        return

    # 3. Esegui le predizioni PER SLIDE
    slide_predictions = predict_slides(model, test_loader, device)

    # 4. Formatta e salva la submission
    submission_df = pd.DataFrame(slide_predictions)
    submission_df['label'] = submission_df['predicted_class_idx'].map(inv_label_map)
    
    # Seleziona e ordina le colonne finali
    final_submission = submission_df[['sample_index', 'label']].sort_values("sample_index")

    output_path = Path(f"./{experiment_id}_submission.csv")
    final_submission.to_csv(output_path, index=False)
    print(f"\n✅ Inference complete. Submission file saved to '{output_path}'")


# =============================================================================
# --- HELPERS DI INFERENZA (PER MODELLO MIL) ---
# =============================================================================

def predict_slides(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> List[Dict]:
    """
    Esegue l'inferenza per ogni SLIDE (sacco) nel DataLoader e restituisce una lista di risultati.
    """
    model.eval()
    slide_results = []

    with torch.no_grad():
        for bags, slide_names in tqdm(loader, desc="Predicting on slides"):
            # Sposta i dati sul device
            bags_on_device = [b.to(device) for b in bags]
            
            # Il modello MIL fa già l'aggregazione e restituisce una previsione per slide
            logits = model(bags_on_device)
            
            # Se un batch non ha prodotto output, saltalo
            if logits.size(0) == 0:
                continue

            # Ottieni le previsioni per il batch
            predictions = logits.argmax(dim=1).cpu().numpy()
            
            # Salva il risultato per ogni slide nel batch
            for i, slide_name in enumerate(slide_names):
                slide_results.append({
                    'sample_index': slide_name,
                    'predicted_class_idx': predictions[i]
                })

    return slide_results

# =============================================================================
# --- Esempio di come chiamare la funzione nel tuo notebook ---
# =============================================================================

# if __name__ == '__main__':
#     # 1. Definisci i parametri
#     MODEL_PATH = Path("./fit_models/AttentionMIL_..._best_model.pth")
#     BASE_DATA_PATH = Path("/path/to/your/preprocessed_data_for_mil") # Assicurati sia il path giusto
#     BATCH_SIZE = 8 # Usa un batch size basso per l'inferenza MIL
#     EXPERIMENT_ID = "mil_inference_vit_01"
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 2. Chiama la funzione principale di inferenza
#     make_mil_inference(
#         model_path=MODEL_PATH,
#         base_path=BASE_DATA_PATH,
#         device=DEVICE,
#         batch_size=BATCH_SIZE,
#         experiment_id=EXPERIMENT_ID,
#         backbone_name='vit_small_patch16_224' # Assicurati che corrisponda al modello addestrato
#     )