import torch
import numpy as np
from sklearn.metrics import f1_score
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda=0, l2_lambda=0, debug_mode=True, comet_experiment=None):
    """
    Perform one complete training epoch through the entire training dataset.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
        scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
        l1_lambda (float): Lambda for L1 regularization
        l2_lambda (float): Lambda for L2 regularization

    Returns:
        tuple: (average_loss, f1 score) - Training loss and f1 score for this epoch
    """
    if debug_mode:
        print(f"[DEBUG] train_one_epoch started", flush=True)
    model.train()  # Set model to training mode

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    # Iterate through training batches of the data loader
    if debug_mode:
        print(f"[DEBUG] Starting batch iteration, total batches: {len(train_loader)}", flush=True)
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle both single-input (inputs, targets) and multi-scale (context, detail, targets) formats
        if len(batch_data) == 3:
            # Multi-scale format: (context, detail, targets)
            context, detail, targets = batch_data
            context, detail, targets = context.to(device), detail.to(device), targets.to(device)
            inputs = (context, detail)  # Tuple for dual-branch model
        else:
            # Standard format: (inputs, targets)
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
        
        if debug_mode and batch_idx == 0:
            if isinstance(inputs, tuple):
                print(f"[DEBUG] Processing first batch, context shape: {inputs[0].shape}, detail shape: {inputs[1].shape}", flush=True)
            else:
                print(f"[DEBUG] Processing first batch, shape: {inputs.shape}", flush=True)
        
        # Clear gradients from previous step
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision (if CUDA available)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # inputs is either a single tensor or a tuple (context, detail) for dual-branch
            logits = model(inputs)
            if debug_mode and batch_idx == 0:
                print(f"[DEBUG] Forward pass done, logits shape: {logits.shape}", flush=True)
            loss = criterion(logits, targets)

            # Add L1 and L2 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm


        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        if debug_mode and batch_idx == 0:
            print(f"[DEBUG] First batch complete", flush=True)
        
        if debug_mode and (batch_idx + 1) % 10 == 0:
            print(f"[DEBUG] Processed {batch_idx + 1}/{len(train_loader)} batches", flush=True)

    if debug_mode:
        print(f"[DEBUG] All batches processed, computing metrics...", flush=True)
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    # ... ottieni preds e labels ...
    cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predictions))
    if debug_mode:
        print("Confusion Matrix:\n", cm)
    comet_experiment.log_confusion_matrix(matrix=cm, labels=[str(i) for i in range(cm.shape[0])], name="Confusion Matrix")

    return epoch_loss, epoch_f1


def validate_one_epoch(model, val_loader, criterion, device, debug_mode=True):
    """
    Perform one complete validation epoch through the entire validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate (must be in eval mode)
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        criterion (nn.Module): Loss function used to calculate validation loss
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)

    Returns:
        tuple: (average_loss, accuracy) - Validation loss and accuracy for this epoch

    Note:
        This function automatically sets the model to evaluation mode and disables
        gradient computation for efficiency during validation.
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    # Disable gradient computation for validation
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both single-input and multi-scale formats
            if len(batch_data) == 3:
                context, detail, targets = batch_data
                context, detail, targets = context.to(device), detail.to(device), targets.to(device)
                inputs = (context, detail)
            else:
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision (if CUDA available)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # inputs is either a single tensor or a tuple (context, detail) for dual-branch
                logits = model(inputs)
                loss = criterion(logits, targets)

            # Accumulate metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_accuracy



def fit(model, train_loader, val_loader, epochs, criterion, optimizer, scaler, device,
        l1_lambda=0, l2_lambda=0, patience=0, evaluation_metric="val_f1", mode='max',
        restore_best_weights=True, writer=None, verbose=10, experiment_name="", comet_experiment=None, debug_mode=True, local_data_path=None):
    """
    Train the neural network model on the training data and validate on the validation data.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        epochs (int): Number of training epochs
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
        scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
        l1_lambda (float): L1 regularization coefficient (default: 0)
        l2_lambda (float): L2 regularization coefficient (default: 0)
        patience (int): Number of epochs to wait for improvement before early stopping (default: 0)
        evaluation_metric (str): Metric to monitor for early stopping (default: "val_f1")
        mode (str): 'max' for maximizing the metric, 'min' for minimizing (default: 'max')
        restore_best_weights (bool): Whether to restore model weights from best epoch (default: True)
        writer (SummaryWriter, optional): TensorBoard SummaryWriter object for logging (default: None)
        verbose (int, optional): Frequency of printing training progress (default: 10)
        experiment_name (str, optional): Experiment name for saving models (default: "")
        comet_experiment : comet experiment variable for logging

    Returns:
        tuple: (model, training_history) - Trained model and metrics history
    """


    if debug_mode:
        print("[DEBUG] fit() function started", flush=True)
    
    # Initialize metrics tracking
    #keeps track of all values during training and validation
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    # Configure early stopping if patience is set
    if patience > 0:
        patience_counter = 0
        best_metric = float('-inf') if mode == 'max' else float('inf')
        best_epoch = 0

    if debug_mode:
        print(f"[DEBUG] Training {epochs} epochs...", flush=True)

    # Main training loop: iterate through epochs
    for epoch in range(1, epochs + 1):
        if debug_mode:
            print(f"Starting epoch {epoch}...", flush=True)  # Debug line
        
        # Forward pass through training data, compute gradients, update weights
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, l1_lambda, l2_lambda, debug_mode=debug_mode, comet_experiment=comet_experiment
        )

        if debug_mode:
            print(f"Epoch {epoch} train done", flush=True)  # Debug line

        # Evaluate model on validation data without updating weights
        val_loss, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device, debug_mode=debug_mode
        )
        # Store metrics for plotting and analysis
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_f1'].append(train_f1)
        training_history['val_f1'].append(val_f1)

        # Write metrics to TensorBoard for visualization
        if writer is not None:
            log_metrics_to_tensorboard(
                writer, epoch, train_loss, train_f1, val_loss, val_f1, model
            )

        # log to comet
        metrics={"train_loss": train_loss, "train_f1": train_f1, "val_loss":val_loss, "val_f1": val_f1 }
        comet_experiment.log_metrics(metrics, step=epoch, epoch=epoch)



        # Print progress every N epochs or on first epoch
        if verbose > 0:
            if epoch % verbose == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: Loss={train_loss:.4f}, F1 Score={train_f1:.4f} | "
                    f"Val: Loss={val_loss:.4f}, F1 Score={val_f1:.4f}")

        # Early stopping logic: monitor metric and save best model
        if patience > 0:
            current_metric = training_history[evaluation_metric][-1]
            is_improvement = (current_metric > best_metric) if mode == 'max' else (current_metric < best_metric)

            fit_models_folder = str(local_data_path)+"/fit_models"
            os.makedirs(fit_models_folder, exist_ok=True)

            if is_improvement:
                best_metric = current_metric
                torch.save(model.state_dict(), fit_models_folder+"/"+experiment_name+'_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    # Restore best model weights if early stopping was used
    if restore_best_weights and patience > 0:
        model.load_state_dict(torch.load(fit_models_folder+"/"+experiment_name+'_model.pt'))
        print(f"Best model restored from epoch {best_epoch} with {evaluation_metric} {best_metric:.4f}")

    # Save final model if no early stopping
    if patience == 0:
        torch.save(model.state_dict(), fit_models_folder+"/"+experiment_name+'_model.pt')

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    comet_experiment.log_model(name="test1", file_or_folder=fit_models_folder+"/"+experiment_name+'_model.pt')

    #comet_experiment.end() 
    return model, training_history


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

