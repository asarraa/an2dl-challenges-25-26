import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm # Assicurati che tqdm sia importato

# =============================================================================
# --- FUNZIONE DI TRAINING PER EPOCA (CORRETTA PER MULTI-SCALA) ---
# =============================================================================import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda=0, l2_lambda=0, debug_mode=True, comet_experiment=None):
    """
    Esegue un'epoca di training per un modello Multiple Instance Learning (MIL).
    """
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    samples_processed = 0

    pbar = tqdm(train_loader, desc=f"Training Epoch", leave=False)
    
    # --- MODIFICA CHIAVE: Spacchetta in 2 elementi (bags, labels) ---
    for bags, labels in pbar:
        # Se il batch è vuoto a causa del filtraggio nella collate_fn, saltalo
        if not bags:
            continue

        labels = labels.to(device)
        # Sposta ogni "sacco" (bag) di tile sul device
        bags_on_device = [b.to(device) for b in bags]
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # Passa la lista di sacchi al modello MIL
            logits = model(bags_on_device)
            
            # Controlla che il modello abbia prodotto output (potrebbe non farlo se tutti i sacchi erano vuoti)
            if logits.size(0) == 0:
                continue
                
            loss = criterion(logits, labels)

            # Aggiungi regolarizzazione (se presente)
            if l1_lambda > 0 or l2_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        num_in_batch = labels.size(0)
        running_loss += loss.item() * num_in_batch
        samples_processed += num_in_batch

        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if samples_processed == 0:
        return 0.0, 0.0

    epoch_loss = running_loss / samples_processed
    all_labels_np = np.concatenate(all_labels)
    all_predictions_np = np.concatenate(all_predictions)
    epoch_f1 = f1_score(all_labels_np, all_predictions_np, average='weighted')

    # (Logica per Comet invariata)
    if comet_experiment:
        cm = confusion_matrix(all_labels_np, all_predictions_np)
        comet_experiment.log_confusion_matrix(matrix=cm, labels=[str(i) for i in range(cm.shape[0])], name="Train Confusion Matrix")

    return epoch_loss, epoch_f1


def validate_one_epoch(model, val_loader, criterion, device, debug_mode=True):
    """
    Esegue un'epoca di validazione per un modello Multiple Instance Learning (MIL).
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    samples_processed = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch", leave=False)
        
        # --- MODIFICA CHIAVE: Spacchetta in 2 elementi (bags, labels) ---
        for bags, labels in pbar:
            if not bags:
                continue

            labels = labels.to(device)
            bags_on_device = [b.to(device) for b in bags]

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(bags_on_device)
                
                if logits.size(0) == 0:
                    continue
                
                loss = criterion(logits, labels)

            num_in_batch = labels.size(0)
            running_loss += loss.item() * num_in_batch
            samples_processed += num_in_batch

            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    if samples_processed == 0:
        print("Warning: Validation set was empty or all batches were skipped.")
        return 0.0, 0.0

    epoch_loss = running_loss / samples_processed
    epoch_f1 = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_f1

# (La funzione di logging per Tensorboard rimane invariata)

# -----------------------------
# Tensorboard
# -----------------------------

def log_metrics_to_tensorboard(writer, epoch, train_loss, train_f1, val_loss, val_f1, model):
    """
    Log training metrics and model parameters to TensorBoard for visualization.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter object for logging
        epoch (int): Current epoch number (used as x-axis in TensorBoard plots)
        train_loss (float): Training loss for this epoch
        train_f1 (float): Training f1 score for this epoch
        val_loss (float): Validation loss for this epoch
        val_f1 (float): Validation f1 score for this epoch
        model (nn.Module): The neural network model (for logging weights/gradients)

    Note:
        This function logs scalar metrics (loss/f1 score) and histograms of model
        parameters and gradients, which helps monitor training progress and detect
        issues like vanishing/exploding gradients.
    """
    # Log scalar metrics
    writer.add_scalar('Loss/Training', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('F1/Training', train_f1, epoch)
    writer.add_scalar('F1/Validation', val_f1, epoch)

    # Log model parameters and gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if the tensor is not empty before adding a histogram
            if param.numel() > 0:
                writer.add_histogram(f'{name}/weights', param.data, epoch)
            if param.grad is not None:
                # Check if the gradient tensor is not empty before adding a histogram
                if param.grad.numel() > 0:
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        writer.add_histogram(f'{name}/gradients', param.grad.data, epoch)

