import config
import torch
import pandas as pd
from lazy_loaders import _resolve_paths, get_test_loaders
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import models
from torch.utils.data import DataLoader 
import numpy as np
import torch.nn.functional as F


def make_inference(loader, device, input_shape, model_path, model_name, experiment_id, base_path, inference_type="weighted_majority_vote"):
    """
    Esegue inferenza e aggregazione.
    inference_type: "weighted_majority_vote" (Media pesata) o "mil" (Max Pooling).
    """
    
    print(f"[INFO] Starting inference with mode: {inference_type}")
    
    inv_label_map = {v: k for k, v in config.LABEL_MAP.items()}

    _, _, csv_path = _resolve_paths(base_path=base_path, add_mask_channel=False, is_test=True)

    # Read CSV metadata into memory
    df_test = pd.read_csv(csv_path)

    if "weight" not in df_test.columns:
        print("[WARNING] Colonna 'weight' non trovata in test_patches.csv. Uso peso 1.0 per tutti.")
        df_test["weight"] = 1.0

    # Preload checkpoint metadata
    ckpt, state_dict = load_checkpoint(model_path, map_location="cpu")
    ckpt_channels = infer_conv1_in_channels(state_dict)

    model = load_model(
        model_path,
        input_shape=input_shape,
        num_classes=len(config.LABEL_MAP),
        model_name=model_name,
        state_dict=state_dict,
        ckpt=ckpt,
        ckpt_channels=ckpt_channels,
    )
    model.to(device)
    
    # 1. Otteniamo le probabilità per ogni tile
    num_classes = len(config.LABEL_MAP)
    tile_preds = predict2(model, loader, device, num_classes)
    prob_cols = [f"prob_{i}" for i in range(num_classes)]

    # 2. Scegliamo la strategia di aggregazione (Bag-level inference)
    if inference_type == "weighted_majority_vote":
        # Strategia attuale: Media pesata delle probabilità
        submission, slide_map = weighted_mean_aggregation(tile_preds, df_test, prob_cols, inv_label_map)
    
    elif inference_type == "mil":
        # Strategia MIL: Max Pooling (il tile più forte decide per la slide)
        submission, slide_map = mil_aggregation(tile_preds, df_test, prob_cols, inv_label_map)
        
    else:
        raise ValueError(f"Inference type '{inference_type}' non supportato. Usa 'weighted_majority_vote' o 'mil'.")

    # Salvataggio
    output = Path(base_path+"/"+experiment_id+"_submission.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"[INFO] Saved submission to {output}")


def load_model(
    model_path: Path,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    model_name: Optional[str],
    state_dict: Optional[dict] = None,
    ckpt: Optional[dict] = None,
    ckpt_channels: Optional[int] = None,
) -> torch.nn.Module:
    ckpt, loaded_state_dict = load_checkpoint(model_path) if state_dict is None else (ckpt, state_dict)

    cfg = {}
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("config"), dict):
            cfg.update(ckpt["config"])
        if isinstance(ckpt.get("model_architecture"), dict):
            cfg.update(ckpt["model_architecture"])

    name = pick_model_name(model_name, ckpt or {}, loaded_state_dict)
    c, h, w = input_shape

    if name == "CNN":
        model_cfg = config.CNN_DEFAULTS.copy()
        model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
        model_cfg["input_shape"] = (c, h, w)
        model_cfg["num_classes"] = num_classes
        model = models.CNN(**model_cfg)
    elif name == "EfficientNet":
        model_cfg = config.EFFICIENTNET_DEFAULTS.copy()
        model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
        model_cfg["input_shape"] = (c, h, w)
        model_cfg["num_classes"] = num_classes
        model = models.EfficientNetModel(**model_cfg)
    elif name == "HistologyResNet":
        if ckpt_channels == 4 or c == 4:
            raise ValueError("[ERROR] Tried using HistologyResNet with 4 input channels, which is obsolete.")
        else:
            model_cfg = config.RESNET_DEFAULTS.copy()
            model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
            model_cfg["num_classes"] = num_classes
            model_cfg["input_channels"] = c
            model = models.HistologyResNet(**model_cfg)
    elif name == "FineTunedResNet50":
        model_cfg = config.RESNET50_FINETUNE_DEFAULTS.copy()
        model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
        model_cfg["num_classes"] = num_classes
        model_cfg["input_channels"] = c
        model = models.FineTunedResNet50(**model_cfg)
    elif name == "HistologyDenseNet":
        model_cfg = config.HISTOLOGY_DENSENET_DEFAULTS.copy()
        model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
        model_cfg["num_classes"] = num_classes
        model_cfg["input_channels"] = c
        model = models.HistologyDenseNet(**model_cfg)
    else:
        raise ValueError(f"Unsupported model '{name}'. Use --model-name to pick a valid one.")

    model.load_state_dict(loaded_state_dict)
    return model


def load_checkpoint(path: Path, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) else {})
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return ckpt, state_dict


def infer_conv1_in_channels(state_dict: dict) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith("conv1.weight") and hasattr(value, "shape"):
            return value.shape[1]
    return None


# ------------------------------
# Inference Helpers
# ------------------------------

def predict2(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> pd.DataFrame:
    """
    Esegue l'inferenza restituendo le probabilità (Softmax) per ogni classe.
    """
    model.eval()
    names = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, batch_names, _ = batch 
            else:
                imgs, batch_names = batch

            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            
            names.extend(batch_names)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    data = {"sample_index": names}
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    for i, col in enumerate(prob_cols):
        data[col] = all_probs[:, i]

    return pd.DataFrame(data)


def weighted_mean_aggregation(
    tile_df: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    prob_cols: List[str], 
    inv_label_map: Dict[int, str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregazione 'Weighted Majority Vote' (Media Pesata delle Probabilità).
    Smussa i picchi e decide in base alla "massa" di evidenza.
    """
    merged = tile_df.merge(meta_df[["sample_index", "original_sample", "weight"]], on="sample_index", how="left")
    
    if merged["weight"].isnull().any():
        merged["weight"] = merged["weight"].fillna(1.0)

    # Numeratore: Probabilità * Peso
    weighted_probs = merged[prob_cols].multiply(merged["weight"], axis=0)
    weighted_probs["original_sample"] = merged["original_sample"]
    
    # Somma per Slide (equivale alla media pesata ai fini dell'argmax)
    grouped_sums = weighted_probs.groupby("original_sample")[prob_cols].sum()
    
    # Argmax
    best_cols = grouped_sums.idxmax(axis=1)
    
    rows = []
    slide_map = {}
    for slide_id, col_name in best_cols.items():
        pred_idx = prob_cols.index(col_name)
        label_str = inv_label_map[pred_idx]
        slide_map[slide_id] = (pred_idx, label_str)
        rows.append({"sample_index": slide_id, "label": label_str})
        
    return pd.DataFrame(rows).sort_values("sample_index"), slide_map


def mil_aggregation(
    tile_df: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    prob_cols: List[str], 
    inv_label_map: Dict[int, str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregazione 'MIL' (Multiple Instance Learning - Max Pooling).
    La classe della slide è determinata dal tile con la probabilità più alta (pesata) per quella classe.
    Cerca il "witness" (testimone) più forte.
    """
    # 1. Merge
    merged = tile_df.merge(meta_df[["sample_index", "original_sample", "weight"]], on="sample_index", how="left")
    
    if merged["weight"].isnull().any():
        merged["weight"] = merged["weight"].fillna(1.0)

    # 2. Applichiamo comunque il peso per filtrare artefatti (es. sfondo nero con peso basso)
    # Se il peso è 0, la probabilità diventa 0 e non verrà selezionata dal Max Pooling.
    weighted_probs = merged[prob_cols].multiply(merged["weight"], axis=0)
    weighted_probs["original_sample"] = merged["original_sample"]
    
    # 3. Max Pooling per Slide: Prendi il valore MASSIMO per ogni classe tra i tile della slide
    # Esempio: Se una slide ha 1000 tile sani e 1 tile tumore (prob=0.9), il max per tumore sarà 0.9.
    grouped_max = weighted_probs.groupby("original_sample")[prob_cols].max()
    
    # 4. Argmax tra le classi basato sui massimi
    best_cols = grouped_max.idxmax(axis=1)
    
    rows = []
    slide_map = {}
    for slide_id, col_name in best_cols.items():
        pred_idx = prob_cols.index(col_name)
        label_str = inv_label_map[pred_idx]
        slide_map[slide_id] = (pred_idx, label_str)
        rows.append({"sample_index": slide_id, "label": label_str})
        
    return pd.DataFrame(rows).sort_values("sample_index"), slide_map


def pick_model_name(user_choice: Optional[str], ckpt: dict, state_dict: dict) -> str:
    if user_choice: return user_choice
    for key in ("model_name",):
        if key in ckpt: return ckpt[key]
    cfg = ckpt.get("config") or {}
    if isinstance(cfg, dict) and "model_name" in cfg: return cfg["model_name"]
    inferred = infer_model_from_state_dict(state_dict)
    if inferred: return inferred
    return config.MODEL_NAME

def infer_model_from_state_dict(state_dict: dict) -> Optional[str]:
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.layer1") or k.startswith("backbone.conv1") for k in keys): return "FineTunedResNet50"
    if any(k.startswith("model.layer1") or k.startswith("model.conv1") for k in keys): return "HistologyResNet"
    if any(k.startswith("features.") for k in keys): return "CNN"
    if any("units.0" in k or "MBConvBlock" in k for k in keys): return "EfficientNet"
    return None


# ------------------------------
# Ensemble Helpers
# ------------------------------

def _run_single_model_inference(
    loader: DataLoader,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    model_path: Path,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs tile-level inference for a single checkpoint and returns probabilities per class.
    """
    ckpt, state_dict = load_checkpoint(model_path, map_location="cpu")
    ckpt_channels = infer_conv1_in_channels(state_dict)
    model = load_model(
        model_path=model_path,
        input_shape=input_shape,
        num_classes=len(config.LABEL_MAP),
        model_name=model_name,
        state_dict=state_dict,
        ckpt=ckpt,
        ckpt_channels=ckpt_channels,
    )
    model.to(device)
    return predict2(model, loader, device, num_classes=len(config.LABEL_MAP))


def aggregate_patch_votes(
    tile_dfs: List[pd.DataFrame],
    vote_strategy: str = "soft",
    model_weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Aggregates per-tile predictions coming from multiple models.
    vote_strategy:
        - 'soft': weighted average of probabilities (default).
        - 'hard': weighted majority vote on argmax, converted back to probabilities.
    """
    if not tile_dfs:
        raise ValueError("No tile predictions provided for aggregation.")

    prob_cols = [c for c in tile_dfs[0].columns if c.startswith("prob_")]
    num_classes = len(prob_cols)
    sample_order = tile_dfs[0]["sample_index"].tolist()

    # Align probabilities across models using the same sample order
    aligned_probs = []
    for idx, df in enumerate(tile_dfs):
        missing = set(sample_order) - set(df["sample_index"])
        if missing:
            raise ValueError(f"Model {idx} is missing {len(missing)} tiles compared to the reference set.")
        aligned = df.set_index("sample_index").loc[sample_order, prob_cols].to_numpy()
        aligned_probs.append(aligned)

    weights = model_weights if model_weights is not None else [1.0] * len(aligned_probs)
    if len(weights) != len(aligned_probs):
        raise ValueError("Length of model_weights must match number of prediction DataFrames.")
    weight_array = np.array(weights, dtype=float)
    weight_array = weight_array / weight_array.sum()

    if vote_strategy == "soft":
        prob_matrix = np.zeros_like(aligned_probs[0])
        for arr, w in zip(aligned_probs, weight_array):
            prob_matrix += arr * w
    elif vote_strategy == "hard":
        votes = np.zeros((len(sample_order), num_classes), dtype=float)
        for arr, w in zip(aligned_probs, weight_array):
            preds = arr.argmax(axis=1)
            votes[np.arange(len(sample_order)), preds] += w
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        prob_matrix = votes / row_sums
    else:
        raise ValueError(f"Unknown vote_strategy '{vote_strategy}'. Use 'soft' or 'hard'.")

    aggregated_df = pd.DataFrame(prob_matrix, columns=prob_cols)
    aggregated_df.insert(0, "sample_index", sample_order)
    aggregated_df["ensemble_pred_idx"] = aggregated_df[prob_cols].values.argmax(axis=1)
    aggregated_df["ensemble_pred_label"] = aggregated_df["ensemble_pred_idx"].map({i: lbl for lbl, i in config.LABEL_MAP.items()})
    return aggregated_df


def make_ensemble_inference(
    loader: DataLoader,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    model_paths: List[Path],
    model_names: Optional[List[Optional[str]]] = None,
    base_path: Optional[str] = None,
    inference_type: str = "weighted_majority_vote",
    patch_vote: str = "soft",
    model_weights: Optional[List[float]] = None,
    save_prefix: str = "ensemble",
):
    """
    Runs inference for multiple checkpoints, aggregates per-tile predictions, and
    produces a slide-level submission via the chosen aggregation method.
    """
    if model_names is None:
        model_names = [None] * len(model_paths)
    if len(model_names) != len(model_paths):
        raise ValueError("model_names length must match model_paths length.")

    inv_label_map = {v: k for k, v in config.LABEL_MAP.items()}
    _, _, csv_path = _resolve_paths(base_path=base_path, add_mask_channel=False, is_test=True)
    meta_df = pd.read_csv(csv_path)
    if "weight" not in meta_df.columns:
        meta_df["weight"] = 1.0

    tile_predictions = []
    for idx, (path, name) in enumerate(zip(model_paths, model_names)):
        print(f"[INFO] Running ensemble member {idx + 1}/{len(model_paths)} -> {path}")
        preds = _run_single_model_inference(
            loader=loader,
            device=device,
            input_shape=input_shape,
            model_path=Path(path),
            model_name=name,
        )
        preds["model_id"] = name or Path(path).stem
        tile_predictions.append(preds)

    aggregated_tiles = aggregate_patch_votes(tile_predictions, vote_strategy=patch_vote, model_weights=model_weights)
    prob_cols = [f"prob_{i}" for i in range(len(config.LABEL_MAP))]

    if inference_type == "weighted_majority_vote":
        submission, slide_map = weighted_mean_aggregation(aggregated_tiles, meta_df, prob_cols, inv_label_map)
    elif inference_type == "mil":
        submission, slide_map = mil_aggregation(aggregated_tiles, meta_df, prob_cols, inv_label_map)
    else:
        raise ValueError(f"Inference type '{inference_type}' non supportato.")

    base = Path(base_path) if base_path is not None else Path(".")
    patch_out = base / f"{save_prefix}_patch_predictions.csv"
    submission_out = base / f"{save_prefix}_submission.csv"
    aggregated_tiles.to_csv(patch_out, index=False)
    submission.to_csv(submission_out, index=False)
    print(f"[INFO] Saved ensemble patch predictions to {patch_out}")
    print(f"[INFO] Saved ensemble submission to {submission_out}")

    return submission, slide_map, aggregated_tiles
