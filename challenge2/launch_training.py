import os
import torch
import torch.nn as nn
import multiscale_pipeline
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from comet_ml import start

# Local Imports
import config
import models
import registry_module
# We don't need preprocessing here because we pass data from the notebook
from training_engine import fit 


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss

def get_device(cfg_device):
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def initialize_training():
    # Initialize best model tracking variables
    best_model = None
    best_performance = float('-inf')
    return best_model, best_performance

def instantiate_model(model_name, current_model_cfg, data_input_shape, device_obj):
    # We unpack (**current_model_cfg) directly into the class
    if model_name == "CNN":
        model = models.CNN(**current_model_cfg)
    elif model_name == "CNNCustom":
        model = models.CNNCustom(**current_model_cfg)
    elif model_name == "EfficientNet":
        model = models.EfficientNetModel(**current_model_cfg)
    
    elif model_name == "HistologyResNet":
        # Rimuoviamo input_shape se è stato aggiunto al config, 
        # perché HistologyResNet non lo accetta nel costruttore __init__
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
            
        model = models.HistologyResNet(**cfg_copy)
        
    elif model_name == "PretrainedEfficientNet":
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
        model = models.PretrainedEfficientNet(**cfg_copy)
        
    elif model_name == "FineTunedResNet50":
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
        model = models.FineTunedResNet50(**cfg_copy)
    
    elif model_name == "FineTunedResNet18":
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
        model = models.FineTunedResNet18(**cfg_copy)
        
    elif model_name == "HistologyDenseNet":
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
        model = models.HistologyDenseNet(**cfg_copy)
        model.to(device_obj)
        return model
    elif model_name == "MultiScale":
        cfg_copy = current_model_cfg.copy()
        if 'input_shape' in cfg_copy:
            del cfg_copy['input_shape']
        model = models.DualBranchResNet(**cfg_copy)
    # Move model to device BEFORE calling summary (torchsummary requires this)
    model = model.to(device_obj)

    # Pass device to torchsummary so it creates input on the correct device
    #summary(model, input_size=data_input_shape, device=str(device_obj.type))
    #model_graph = draw_graph(model, input_size=(batch_size)+config.input_shape, expand_nested=True, depth=5)
    #model_graph.visual_graph
    return model

def get_criterion_from_name(criterion_name, device, class_weights=None):
# Default to CrossEntropy if name matches or if generic "crossentropy" is used
    if criterion_name == "CrossEntropyLoss" or criterion_name == "crossentropy":
        if class_weights is None:
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            return nn.CrossEntropyLoss(weight=class_weights.to(device),label_smoothing=0.1)
    elif criterion_name == "FocalLoss":
        if class_weights is None:
            return FocalLoss(alpha=class_weights, gamma=2.5)
        else:
            return FocalLoss(gamma=2.5)
    else:
        print(f"Warning: Criterion '{criterion_name}' not found. Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss(weight=class_weights.to(device))

def get_optimizer_and_scaler(optimizer_name, model, learning_rate, l2_lambda, device_obj):
    # --- FILTRO PARAMETRI (NUOVO STEP) ---
    # Seleziona solo i parametri che devono essere addestrati (requires_grad=True)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # Define optimizer with L2 regularization
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=l2_lambda)
    else :
        print("ERR! Optimizer not recognized. Using AdamW as default.")
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=l2_lambda)

    # Enable mixed precision training for GPU acceleration
    scaler = torch.amp.GradScaler(enabled=(device_obj.type == 'cuda'))
    return optimizer, scaler

def start_training(model_name="CNN", model_params=None, training_params=None, device=None, data_input_shape=None, debug_mode=False, local_data_path=None, class_weights=None, data_path=None, batch_size=128):
    """
    Args:
        model_name (str): "CNN" or "EfficientNet"
        model_params (dict): Dictionary of overrides for the model architecture.
        training_params (dict): Dictionary of overrides for training (lr, epochs, etc).
        data_path (str or Path): Path to the preprocessed data folder. Required.
    """
    from pathlib import Path
    
    # Validate data_path is provided
    if data_path is None:
        raise ValueError(
            "data_path is required! Please provide the path to your preprocessed data folder.\n"
            "Example: data_path='/kaggle/working/an2dl-challenges-25-26/challenge2/data/preprocessed/preprocess_v1'"
        )
    
    # Convert to Path if string
    data_path = Path(data_path)

    train_loader, val_loader, _, class_weights = multiscale_pipeline.get_multiscale_loaders(
        base_path=data_path,
        batch_size=batch_size
    )
    
    reg_manager = registry_module.ModelRegistry(local_data_path)
    run_id = reg_manager.generate_id(prefix=model_name)

    best_model, best_performance = initialize_training()

    # Handle device: accept both string ("cuda"/"cpu") or torch.device object
    if isinstance(device, torch.device):
        device_obj = device
    elif device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    elif device == "cuda":
        print("WARNING: CUDA requested but not available! Using CPU instead.", flush=True)
        device_obj = torch.device("cpu")
    elif device == "cpu":
        device_obj = torch.device("cpu")
    else:
        # Default: use CUDA if available, otherwise CPU
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if debug_mode:
        print(f"[DEBUG] Device type: {type(device)}, Final device: {device_obj}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if device_obj.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: Using CPU (this will be slow!)", flush=True)
    print(f"--- Starting {model_name} on {device_obj} ---", flush=True)
    # -------------------------------------------------------
    # 1. SETUP CONFIGURATION (Merge Defaults + Overrides)
    # -------------------------------------------------------
    
    # A. Prepare Training Config
    # Start with defaults from config.py
    current_train_cfg = config.TRAINING_DEFAULTS.copy()
    current_model_cfg = {}
    # B. Prepare Model Config
    if model_name == "CNN":
        current_model_cfg = config.CNN_DEFAULTS.copy()
    elif model_name == "CNNCustom":
        current_model_cfg = config.CNN_DEFAULTS.copy()
    elif model_name == "EfficientNet":
        current_model_cfg = config.EFFICIENTNET_DEFAULTS.copy()
    elif model_name == "HistologyResNet":
        # Se hai messo RESNET_DEFAULTS in config.py usa quello, 
        # altrimenti definiscilo qui al volo:
        if hasattr(config, 'RESNET_DEFAULTS'):
            current_model_cfg = config.RESNET_DEFAULTS.copy()
        else:
            current_model_cfg = {
                "num_classes": 4, 
                "use_pretrained": True, 
                "backbone": "resnet18"
            }
    elif model_name == "PretrainedEfficientNet":
        current_model_cfg = config.PRETRAINED_EFFICIENTNET_DEFAULTS.copy()
    elif model_name == "FineTunedResNet50":
        current_model_cfg = config.RESNET50_FINETUNE_DEFAULTS.copy()
    elif model_name == "FineTunedResNet18":
        current_model_cfg = config.RESNET18_FINETUNE_DEFAULTS.copy()
    elif model_name == "HistologyDenseNet":
        current_model_cfg = config.HISTOLOGY_DENSENET_DEFAULTS.copy()
    elif model_name == "MultiScale":
        current_model_cfg = config.MULTISCALE_DEFAULTS.copy()
        
    # Update with whatever you passed in (if anything)
    if training_params:
        current_train_cfg.update(training_params)
    # Update with model overrides
    if model_params:
        current_model_cfg.update(model_params)
    
    # Override input_shape with the actual data shape passed as parameter
    if data_input_shape is not None:
        print("Checkpoint")
        print(f"✓ Original input_shape in config: {current_model_cfg.get('input_shape', 'Not set')}")
        current_model_cfg['input_shape'] = data_input_shape
        print(f"✓ Updated input_shape to: {data_input_shape}")

    train_parameters_summary = "\n".join([f"{k}: {v}" for k, v in current_train_cfg.items()])
    model_parameters_summary = "\n".join([f"{k}: {v}" for k, v in current_model_cfg.items()])
    print(f"Starting {model_name} model training...")
    print("Training Configuration:\n", train_parameters_summary)
    print("Model Configuration:\n", model_parameters_summary)

    #Initialize Comet logging
    comet_experiment = start(
      api_key="nhvfD4vUpZNMoJQ3dEjOwIeua",
      project_name="test",
      workspace="asarraa",
    )
    
    comet_experiment.set_name(run_id)

    comet_experiment.log_parameters(current_train_cfg)
    comet_experiment.log_parameters(current_model_cfg)

    # -------------------------------------------------------
    # 2. INSTANTIATE (Using the merged configs)
    # -------------------------------------------------------
    if debug_mode:
        print("[DEBUG] About to instantiate model...", flush=True)
    # Instantiate Model
    model = instantiate_model(model_name, current_model_cfg, data_input_shape, device_obj)
    if debug_mode:
        print("[DEBUG] Model instantiated successfully", flush=True) 
    
    # Get criterion
    criterion = get_criterion_from_name(current_train_cfg['criterion_name'], device_obj, class_weights)

    optimizer, scaler = get_optimizer_and_scaler(current_train_cfg['optimizer_name'], model, current_train_cfg['learning_rate'], current_train_cfg['l2_lambda'], device_obj)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_train_cfg['epochs'])

    # TensorBoard
    writer = SummaryWriter(f"tensorboard/{run_id}")
    '''
    writer = SummaryWriter(f"tensorboard/{experiment_name}")
    x = torch.randn(1, data_input_shape[0], data_input_shape[1], data_input_shape[2]).to(device_obj)
    writer.add_graph(model, x)
    '''

    try:
        if data_input_shape is not None:
            # Crea input dummy sullo stesso device del modello
            x = torch.randn(1, *data_input_shape).to(device_obj)
            writer.add_graph(model, x)
    except Exception as e:
        print(f"Warning: TensorBoard Graph logging failed (skipping): {e}")

    # -------------------------------------------------------
    # 3. RUN TRAINING
    # -------------------------------------------------------

    if debug_mode:
        print("[DEBUG] About to call fit()...", flush=True)
    # Train model and track training history
    model, training_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=current_train_cfg['epochs'],
        criterion=criterion,
        optimizer=optimizer, 
        scaler=scaler,
        device=device_obj, 
        writer=writer,
        l1_lambda=current_train_cfg['l1_lambda'],
        l2_lambda=0, #already applied in AdamW optimizer, don't change!
        verbose=current_train_cfg['verbose'],
        experiment_name=run_id, 
        patience=current_train_cfg['patience'],
        comet_experiment=comet_experiment,
        debug_mode=debug_mode,
        local_data_path=local_data_path,
        scheduler=scheduler,
        )

    # Update best model if current performance is superior
    if training_history['val_f1'][-1] > best_performance:
        best_model = model
        best_performance = training_history['val_f1'][-1]

    # -------------------------------------------------------
    # 5. SAVE TO REGISTRY 
    # -------------------------------------------------------
        
    # Extract final metrics from history
    # We take the last value of the validation F1 and Loss
    final_metrics = {
        "val_f1": training_history['val_f1'][-1],
        "val_loss": training_history['val_loss'][-1],
        "train_loss": training_history['train_loss'][-1],
        "best_val_f1": max(training_history['val_f1']), # or however you track best
        "best_train_f1": max(training_history['train_f1'])

    }
    
    # Add 'model_name' to model_cfg so it appears in the ID
    current_model_cfg["model_name"] = model_name

    # Save everything
    exp_id = reg_manager.save_experiment(
        model=model,
        optimizer=optimizer,
        train_cfg=current_train_cfg,
        model_cfg=current_model_cfg,
        metrics=final_metrics,
        run_id=run_id
    )
    
    # Log the ID to Comet so you can link them
    comet_experiment.log_other("local_experiment_id", exp_id)
    comet_experiment.end()
    
    return model, training_history, exp_id
