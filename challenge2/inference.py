import config
import torch
import pandas as pd
from lazy_loaders import _resolve_paths, get_test_loaders
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import models
from torch.utils.data import DataLoader 



def make_inference(loader, device, input_shape, model_path, model_name, base_path, output="submission.csv"):
    
    inv_label_map = {v: k for k, v in config.LABEL_MAP.items()}

    _, _, csv_path = _resolve_paths(base_path=base_path, add_mask_channel=False, is_test=True)

    # Read CSV metadata into memory (very small footprint)
    df_test = pd.read_csv(csv_path)


    # Preload checkpoint metadata to adapt channels
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

    tile_preds = predict(model, loader, device)

    submission, slide_map = majority_vote(tile_preds, df_test, inv_label_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(output)
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
        backbone = cfg.get("backbone", "resnet18")
        use_pretrained = cfg.get("use_pretrained", False)
        if ckpt_channels == 4 or c == 4:
            raise ValueError("[ERROR] Tried using HistologyResNet with 4 input channels, which is obsolete and not supported.")
            #model = LegacyHistologyResNet(num_classes=num_classes, use_pretrained=use_pretrained, backbone=backbone)
        else:
            model_cfg = config.RESNET_DEFAULTS.copy()
            model_cfg.update({k: v for k, v in cfg.items() if k in model_cfg})
            model_cfg["num_classes"] = num_classes
            model_cfg["input_channels"] = c
            model = models.HistologyResNet(**model_cfg)
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
    """Infer expected input channels from checkpoint weights."""
    for key, value in state_dict.items():
        if key.endswith("conv1.weight") and hasattr(value, "shape"):
            return value.shape[1]
    return None




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
    df = tile_df.copy()
    default_weights = df["weight"] if "weight" in df.columns else 1.0

    # Prefer patch weights coming from the metadata CSV (precomputed during preprocessing)
    if "weight" in meta_df.columns:
        weight_map = meta_df.set_index("sample_index")["weight"]
        df["weight"] = df["sample_index"].map(weight_map).fillna(default_weights)
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    if "original_sample" in meta_df.columns:
        original_map = meta_df.set_index("sample_index")["original_sample"]
        df["original_sample"] = df["sample_index"].map(original_map).fillna(df["sample_index"])
        groups = df.groupby("original_sample")
    else:
        groups = df.groupby("sample_index")

    rows = []
    slide_map = {}
    for slide, group in groups:
        scores = group.groupby("pred_idx")["weight"].sum()
        best_idx = scores.idxmax()
        label_str = inv_label_map[int(best_idx)]
        slide_map[slide] = (int(best_idx), label_str)
        rows.append({"sample_index": slide, "label": label_str})
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
    if "weight" in meta_df.columns:
        weight_map = meta_df.set_index("sample_index")["weight"]
        merged["weight"] = merged["sample_index"].map(weight_map).fillna(merged["weight"] if "weight" in merged.columns else 1.0)
    elif "weight" not in merged.columns:
        merged["weight"] = 1.0
    if "black_ratio" not in merged.columns:
        merged["black_ratio"] = None
    merged["final_pred_idx"] = merged["original_sample"].map(lambda s: slide_map[s][0])
    merged["final_pred_label"] = merged["original_sample"].map(lambda s: slide_map[s][1])
    return merged



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



# ------------------------------
# Model helpers
# ------------------------------
def infer_model_from_state_dict(state_dict: dict) -> Optional[str]:
    keys = list(state_dict.keys())
    if any(k.startswith("model.layer1") or k.startswith("model.conv1") for k in keys):
        return "HistologyResNet"
    if any(k.startswith("features.") for k in keys):
        return "CNN"
    # Simple heuristic for EfficientNetModel keys
    if any("units.0" in k or "MBConvBlock" in k for k in keys):
        return "EfficientNet"
    return None

# i want to try if it works now
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, obs_input_shape = get_test_loaders(batch_size=128, base_path="./data/preprocessed/preprocess_v1_weighted")
    
    # Opzionale: puoi usare obs_input_shape invece di hardcodare (3, 224, 224)
    print(f"Detected input shape: {obs_input_shape}")
    make_inference(
        loader=test_loader,
        device=device,
        input_shape=obs_input_shape,
        model_path=Path("./HistologyResNet_20251210_101039.pt"),
        model_name="HistologyResNet",
        batch_size=128,
        output="submission.csv"
    )