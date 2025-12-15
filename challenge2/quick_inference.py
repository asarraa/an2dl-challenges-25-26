"""
Quick inference + majority voting tailored to this repo layout.

Expected structure (already in data/preprocessed/preprocess_v1):
    preprocess_v1/
        train/train_patches.csv
        test/test_patches.csv
        train/images/*.png
        test/images/*.png

Usage:
    python quick_inference.py --model-path path/to/model.pt \
        [--data-root ./data/preprocessed/preprocess_v1] \
        [--model-name CNN|EfficientNet|HistologyResNet] \
        [--batch-size 128] [--device cuda|cpu] \
        [--output submission.csv]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

import numpy as np
import torch.nn.functional as F

import config
import models

ROOT_DIR = Path(__file__).resolve().parent


# ------------------------------
# Legacy 4-channel ResNet used in earlier trainings
# ------------------------------
class LegacyHistologyResNet(nn.Module):
    """
    ResNet backbone with fixed 4 input channels (RGB + mask).
    Initializes the 4th channel as the mean of pretrained RGB weights.
    """

    def __init__(self, num_classes=4, use_pretrained=True, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.model = tv_models.resnet18(weights="DEFAULT" if use_pretrained else None)
            last_channel_in = self.model.fc.in_features
        elif backbone == "resnet50":
            self.model = tv_models.resnet50(weights="DEFAULT" if use_pretrained else None)
            last_channel_in = self.model.fc.in_features
        else:
            raise ValueError("Backbone supportata: resnet18, resnet50")

        original_conv1 = self.model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        if use_pretrained:
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                new_conv1.weight[:, 3, :, :] = torch.mean(original_conv1.weight, dim=1)

        self.model.conv1 = new_conv1
        self.model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(last_channel_in, num_classes))

    def forward(self, x):
        return self.model(x)


# ------------------------------
# Dataset / Loader helpers
# ------------------------------
class TestDataset(Dataset):
    """Lazy test dataset that keeps only filenames in memory."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        target_channels: Optional[int] = None,
    ):
        self.filenames = df["sample_index"].tolist()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir and masks_dir.exists() else None
        self.add_mask = self.masks_dir is not None
        self.target_channels = target_channels
        self.to_tensor = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.images_dir / name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute black ratio (percentage of very dark pixels) to weight votes
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        black_ratio = (gray < 15).mean().item()

        if self.add_mask:
            mask_path = self.masks_dir / name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load mask {mask_path}")
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image)
            mask_t = torch.from_numpy(mask).unsqueeze(-1)
            image = torch.cat([image, mask_t], dim=-1).numpy()

        image_tensor = self.to_tensor(Image.fromarray(image))
        if self.target_channels:
            c = image_tensor.shape[0]
            if self.target_channels > c:
                pad = torch.zeros(self.target_channels - c, *image_tensor.shape[1:], dtype=image_tensor.dtype)
                image_tensor = torch.cat([image_tensor, pad], dim=0)
            elif self.target_channels < c:
                image_tensor = image_tensor[: self.target_channels]
        return image_tensor, name, float(black_ratio)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: Optional[int] = None) -> DataLoader:
    if workers is None:
        workers = max(2, min(8, os.cpu_count() or 2))
    use_multiprocessing = workers > 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4 if use_multiprocessing else None,
        persistent_workers=True if use_multiprocessing else False,
    )


# ------------------------------
# Model helpers
# ------------------------------
def infer_model_from_state_dict(state_dict: dict) -> Optional[str]:
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.layer1") or k.startswith("backbone.conv1") for k in keys):
        return "FineTunedResNet50"
    if any(k.startswith("model.layer1") or k.startswith("model.conv1") for k in keys):
        return "HistologyResNet"
    if any(k.startswith("features.") for k in keys):
        return "CNN"
    # Simple heuristic for EfficientNetModel keys
    if any("units.0" in k or "MBConvBlock" in k for k in keys):
        return "EfficientNet"
    return None


def pick_model_name(user_choice: Optional[str], ckpt: dict, state_dict: dict) -> str:
    if user_choice:
        return user_choice
    for key in ("model_name",):
        if key in ckpt:
            return ckpt[key]
    cfg = ckpt.get("config") or {}
    if isinstance(cfg, dict) and "model_name" in cfg:
        return cfg["model_name"]

    inferred = infer_model_from_state_dict(state_dict)
    if inferred:
        print(f"[INFO] Detected architecture from checkpoint: {inferred}")
        return inferred

    return config.MODEL_NAME


def load_checkpoint(path: Path, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) else {})
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return ckpt, state_dict


def infer_conv1_in_channels(state_dict: dict) -> Optional[int]:
    """Infer expected input channels from checkpoint weights."""
    for key, value in state_dict.items():
        if key.endswith("conv1.weight") and hasattr(value, "shape"):
            return value.shape[1]
    return None


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
        backbone = cfg.get("backbone", "resnet18")
        use_pretrained = cfg.get("use_pretrained", False)
        if ckpt_channels == 4 or c == 4:
            print("[INFO] Using LegacyHistologyResNet with 4 input channels")
            model = LegacyHistologyResNet(num_classes=num_classes, use_pretrained=use_pretrained, backbone=backbone)
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
    else:
        raise ValueError(f"Unsupported model '{name}'. Use --model-name to pick a valid one.")

    model.load_state_dict(loaded_state_dict)
    return model


# ------------------------------
# Inference + aggregation
# ------------------------------
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    names, preds, weights, black_ratios = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, batch_names, batch_black = batch
            else:
                imgs, batch_names = batch
                batch_black = torch.zeros(len(batch_names))

            imgs = imgs.to(device)
            logits = model(imgs)
            batch_preds = logits.argmax(dim=1).cpu().tolist()

            names.extend(batch_names)
            preds.extend(batch_preds)
            black_list = batch_black.tolist()
            black_ratios.extend(black_list)
            weights.extend([max(0.05, 1.0 - b) for b in black_list])

    return pd.DataFrame(
        {
            "sample_index": names,
            "pred_idx": preds,
            "black_ratio": black_ratios,
            "weight": weights,
        }
    )


def majority_vote(tile_df: pd.DataFrame, meta_df: pd.DataFrame, inv_label_map: Dict[int, str]) -> pd.DataFrame:
    if "weight" not in tile_df.columns:
        tile_df = tile_df.copy()
        tile_df["weight"] = 1.0

    if "original_sample" in meta_df.columns:
        merged = tile_df.merge(meta_df[["sample_index", "original_sample"]], on="sample_index", how="left")
        groups = merged.groupby("original_sample")
    else:
        groups = tile_df.groupby("sample_index")

    rows = []
    slide_map = {}
    for slide, group in groups:
        scores = group.groupby("pred_idx")["weight"].sum()
        best_idx = scores.idxmax()
        label_str = inv_label_map[int(best_idx)]
        slide_map[slide] = (int(best_idx), label_str)
        rows.append({"sample_index": slide, "label": label_str})
    return pd.DataFrame(rows).sort_values("sample_index"), slide_map

# ------------------------------
# Test on probabilities instead of argmax
# ------------------------------

def predict2(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> pd.DataFrame:
    """
    Esegue l'inferenza restituendo le probabilità (Softmax) per ogni classe.
    """
    model.eval()
    names = []
    # Pre-allochiamo una lista per le probabilità
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, batch_names, _ = batch # Ignoriamo black_ratio qui
            else:
                imgs, batch_names = batch

            imgs = imgs.to(device)
            logits = model(imgs)
            
            # Calcola le probabilità con Softmax
            probs = F.softmax(logits, dim=1)
            
            names.extend(batch_names)
            all_probs.append(probs.cpu().numpy())

    # Concatena tutti i batch di probabilità
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Crea il dizionario per il DataFrame
    data = {"sample_index": names}
    
    # Aggiunge colonne dinamicamente: prob_0, prob_1, etc.
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
    Aggrega usando la media ponderata delle probabilità.
    tile_df: contiene le probabilità (output di predict).
    meta_df: contiene 'original_sample' e 'weight' (dal CSV di test).
    """
    
    # 1. Merge: Uniamo le predizioni (tile_df) con i metadati e i pesi (meta_df)
    # Assumiamo che meta_df abbia le colonne 'sample_index', 'original_sample' e 'weight'
    merged = tile_df.merge(meta_df[["sample_index", "original_sample", "weight"]], on="sample_index", how="left")
    
    # Gestione sicurezza: se manca il peso, usa 1.0
    if merged["weight"].isnull().any():
        print("[WARNING] Alcuni pesi sono NaN dopo il merge. Verranno impostati a 1.0")
        merged["weight"] = merged["weight"].fillna(1.0)

    # 2. Calcolo Numeratore: Probabilità * Peso Tile
    weighted_probs = merged[prob_cols].multiply(merged["weight"], axis=0)
    weighted_probs["original_sample"] = merged["original_sample"]
    
    # 3. Somma per Slide (Groupby)
    # Nota: Non serve dividere per la somma dei pesi per trovare l'argmax, 
    # perché il denominatore sarebbe lo stesso per tutte le classi della stessa slide.
    grouped_sums = weighted_probs.groupby("original_sample")[prob_cols].sum()
    
    # 4. Argmax: Trova la classe con il valore cumulativo più alto
    best_cols = grouped_sums.idxmax(axis=1)
    
    rows = []
    slide_map = {}
    
    for slide_id, col_name in best_cols.items():
        # Estrae l'indice numerico dal nome della colonna (es. "prob_3" -> 3)
        # Assicurati che prob_cols sia ordinato correttamente
        pred_idx = prob_cols.index(col_name)
        label_str = inv_label_map[pred_idx]
        
        slide_map[slide_id] = (pred_idx, label_str)
        rows.append({"sample_index": slide_id, "label": label_str})
        
    return pd.DataFrame(rows).sort_values("sample_index"), slide_map

def build_debug_table(tile_df: pd.DataFrame, meta_df: pd.DataFrame, inv_label_map: Dict[int, str], slide_map: Dict[str, tuple]) -> pd.DataFrame:
    """
    Return a dataframe with per-patch prediction and the final slide-level majority vote.
    Columns: sample_index (patch), original_sample, pred_idx, pred_label, weight, black_ratio, final_pred_idx, final_pred_label
    """
    if "original_sample" in meta_df.columns:
        merged = tile_df.merge(meta_df[["sample_index", "original_sample"]], on="sample_index", how="left")
    else:
        merged = tile_df.copy()
        merged["original_sample"] = merged["sample_index"]

    merged["pred_label"] = merged["pred_idx"].map(inv_label_map)
    if "weight" not in merged.columns:
        merged["weight"] = 1.0
    if "black_ratio" not in merged.columns:
        merged["black_ratio"] = None
    merged["final_pred_idx"] = merged["original_sample"].map(lambda s: slide_map[s][0])
    merged["final_pred_label"] = merged["original_sample"].map(lambda s: slide_map[s][1])
    return merged


# ------------------------------
# Paths / CLI
# ------------------------------
def resolve_data_root(user_path: Optional[str]) -> Path:
    if user_path:
        cand = Path(user_path).expanduser().resolve()
    else:
        cand = (ROOT_DIR / "data" / "preprocessed" / "preprocess_v1").resolve()
    if not cand.exists():
        raise FileNotFoundError(f"Dataset not found at {cand}. Pass --data-root explicitly.")
    expected = [cand / "test" / "test_patches.csv", cand / "train" / "train_patches.csv"]
    for p in expected:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")
    return cand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast inference with majority voting.")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to the .pt checkpoint.")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder with train/test subfolders and *_patches.csv.")
    parser.add_argument("--model-name", type=str, default=None, help="Override model type (CNN, EfficientNet, HistologyResNet).")
    parser.add_argument("--batch-size", type=int, default=config.LOADER_PARAMS["batch_size"], help="Batch size for inference.")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (use 0 on macOS if shm issues).")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to auto.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Path for the aggregated submission.")
    parser.add_argument("--save-tiles", type=Path, default=None, help="Optional path to store patch-level predictions.")
    parser.add_argument("--save-debug", type=Path, default=None, help="Optional path to store patch preds + final majority label.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preload checkpoint metadata to adapt channels
    ckpt, state_dict = load_checkpoint(args.model_path, map_location="cpu")
    ckpt_channels = infer_conv1_in_channels(state_dict)

    data_root = resolve_data_root(args.data_root)
    test_csv = data_root / "test" / "test_patches.csv"
    images_dir = data_root / "test" / "images"
    masks_dir = data_root / "test" / "masks"
    add_mask_channel = masks_dir.exists()

    df_test = pd.read_csv(test_csv)
    
    # TESTING
    if "weight" not in df_test.columns:
        print("[WARNING] Colonna 'weight' non trovata in test_patches.csv. Uso peso 1.0 per tutti.")
        df_test["weight"] = 1.0
    
    label_map = config.LABEL_MAP
    inv_label_map = {v: k for k, v in label_map.items()}

    # Peek first image for input shape
    sample_img = cv2.imread(str(images_dir / df_test.iloc[0]["sample_index"]))
    if sample_img is None:
        raise FileNotFoundError(f"Cannot read sample image {images_dir / df_test.iloc[0]['sample_index']}")
    base_channels = sample_img.shape[2] + (1 if add_mask_channel else 0)
    target_channels = ckpt_channels or base_channels
    input_shape = (target_channels, sample_img.shape[0], sample_img.shape[1])

    ds = TestDataset(
        df_test,
        images_dir,
        masks_dir=masks_dir if add_mask_channel else None,
        target_channels=target_channels,
    )
    batch_size = args.batch_size if not add_mask_channel else max(1, args.batch_size // 2)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, workers=args.workers)

    model = load_model(
        args.model_path,
        input_shape=input_shape,
        num_classes=len(label_map),
        model_name=args.model_name,
        state_dict=state_dict,
        ckpt=ckpt,
        ckpt_channels=ckpt_channels,
    )
    model.to(device)

    print(f"[INFO] Inference on {len(ds)} patches | batch_size={batch_size} | device={device}")
    
    # TESTING
    num_classes = len(label_map)
    tile_preds = predict2(model, loader, device, num_classes)
    
    #tile_preds = predict(model, loader, device)

    if args.save_tiles:
        args.save_tiles.parent.mkdir(parents=True, exist_ok=True)
        tile_preds.to_csv(args.save_tiles, index=False)
        print(f"[INFO] Saved patch-level predictions to {args.save_tiles}")
        
    # TESTING
    # Definisci le colonne delle probabilità attese
    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    # 2. Aggregazione pesata usando i pesi presenti in df_test
    submission, slide_map = weighted_mean_aggregation(tile_preds, df_test, prob_cols, inv_label_map)

    #submission, slide_map = majority_vote(tile_preds, df_test, inv_label_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"[INFO] Saved submission to {args.output}")

    if args.save_debug:
        debug_df = build_debug_table(tile_preds, df_test, inv_label_map, slide_map)
        args.save_debug.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(args.save_debug, index=False)
        print(f"[INFO] Saved per-patch debug table to {args.save_debug}")


if __name__ == "__main__":
    main()
