
# 1. IMPORTS
import os
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from comet_ml import start

# Local Imports
import config
import models
import registry_module
# We don't need preprocessing here because we pass data from the notebook
from training_engine import fit 


# -----------------------------
# Helper Functions
# -----------------------------

def get_device(cfg_device):
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initialize_training():
    # Initialize best model tracking variables
    best_model = None
    best_performance = float('-inf')
    return best_model, best_performance

def instantiate_model(model_name, batch_size, current_model_cfg, data_input_shape, device_obj):

    # We unpack (**current_model_cfg) directly into the class
    if model_name == "CNN":
        model = models.CNN(**current_model_cfg)
    elif model_name == "EfficientNet":
        model = models.EfficientNetModel(**current_model_cfg)

    # Move model to device BEFORE calling summary (torchsummary requires this)
    model = model.to(device_obj)

    # Pass device to torchsummary so it creates input on the correct device
    summary(model, input_size=data_input_shape, device=str(device_obj.type))
    #model_graph = draw_graph(model, input_size=(batch_size)+config.input_shape, expand_nested=True, depth=5)
    #model_graph.visual_graph
    return model


def get_criterion_from_name(criterion_name):
# Default to CrossEntropy if name matches or if generic "crossentropy" is used
    if criterion_name == "CrossEntropyLoss" or criterion_name == "crossentropy":
        return nn.CrossEntropyLoss()
    else:
        print(f"Warning: Criterion '{criterion_name}' not found. Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss()
    

def get_optimizer_and_scaler(optimizer_name, model, learning_rate, l2_lambda, device_obj):
    # Define optimizer with L2 regularization
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    else :
        print("ERR! Optimizer not recognized. Using AdamW as default.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Enable mixed precision training for GPU acceleration
    scaler = torch.amp.GradScaler(enabled=(device_obj.type == 'cuda'))
    return optimizer, scaler


def start_training(model_name="CNN", model_params=None, training_params=None, device=None, train_loader=None, val_loader=None, data_input_shape=None):
    """
    Args:
        model_name (str): "CNN" or "EfficientNet"
        model_params (dict): Dictionary of overrides for the model architecture.
        training_params (dict): Dictionary of overrides for training (lr, epochs, etc).
    """

    best_model, best_performance = initialize_training()

    # Ensure 'device' is a proper torch object, not just a string
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device("cpu")
        
    print(f"--- Starting {model_name} on {device_obj} ---")

    # -------------------------------------------------------
    # 1. SETUP CONFIGURATION (Merge Defaults + Overrides)
    # -------------------------------------------------------
    
    # A. Prepare Training Config
    # Start with defaults from config.py
    current_train_cfg = config.TRAINING_DEFAULTS.copy()
    # Update with whatever you passed in (if anything)
    if training_params:
        current_train_cfg.update(training_params)

    # B. Prepare Model Config
    if model_name == "CNN":
        current_model_cfg = config.CNN_DEFAULTS.copy()
    elif model_name == "EfficientNet":
        current_model_cfg = config.EFFICIENTNET_DEFAULTS.copy()
    
    # Update with model overrides
    if model_params:
        current_model_cfg.update(model_params)

    train_parameters_summary = "\n".join([f"{k}: {v}" for k, v in current_train_cfg.items()])
    model_parameters_summary = "\n".join([f"{k}: {v}" for k, v in current_model_cfg.items()])
    print(f"Starting {model_name} model training...")
    print("Training Configuration:\n", train_parameters_summary)
    print("Model Configuration:\n", model_parameters_summary)

    experiment_name = model_name + " parameters:" #TODO




        #Initialize Comet logging
    comet_experiment = start(
      api_key="nhvfD4vUpZNMoJQ3dEjOwIeua",
      project_name="test",
      workspace="asarraa"
    )

    hyper_params = {
      "learning_rate": current_train_cfg['learning_rate'],
      "batch_size": current_train_cfg['batch_size'],
      "epochs": current_train_cfg['epochs'],
      "model": model_name,
    }

    data_input_shape =  data_input_shape #TODO: take it from preprocessing: preprocessing.get_data_input_shape()


    # -------------------------------------------------------
    # 2. INSTANTIATE (Using the merged configs)
    # -------------------------------------------------------

    print("[DEBUG] About to instantiate model...", flush=True)
    # Instantiate Model
    model = instantiate_model(model_name, current_train_cfg['batch_size'], current_model_cfg, data_input_shape, device_obj)
    print("[DEBUG] Model instantiated successfully", flush=True)        
    #model = model.to(device_obj) 
    # Get criterion
    criterion = get_criterion_from_name(current_train_cfg['criterion_name'])  

    optimizer, scaler = get_optimizer_and_scaler(current_train_cfg['optimizer_name'], model, current_train_cfg['learning_rate'], current_train_cfg['l2_lambda'], device_obj)

    # Get data loader
    train_loader = train_loader #TODO: take it from preprocessing
    val_loader = val_loader  #TODO: take it from preprocessing:     preprocessing.get_data_loaders()

    # TensorBoard
    experiment_name = f"{model_name}_run"
    writer = SummaryWriter(f"tensorboard/{experiment_name}")
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
        print(f"⚠️ Warning: TensorBoard Graph logging failed (skipping): {e}")

    # -------------------------------------------------------
    # 3. RUN TRAINING
    # -------------------------------------------------------

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
        experiment_name=experiment_name, #TODO:check names
        patience=current_train_cfg['patience'],
        comet_experiment=comet_experiment
        )

    # Update best model if current performance is superior
    if training_history['val_f1'][-1] > best_performance:
        best_model = model
        best_performance = training_history['val_f1'][-1]

    comet_experiment.log_parameters(hyper_params)


    # -------------------------------------------------------
    # 5. SAVE TO REGISTRY (New Section)
    # -------------------------------------------------------
    
    # Initialize Registry (saves to 'experiments/' folder by default)
    registry = registry.ModelRegistry(base_dir="experiments")
    
    # Extract final metrics from history
    # We take the last value of the validation F1 and Loss
    final_metrics = {
        "val_f1": training_history['val_f1'][-1],
        "val_loss": training_history['val_loss'][-1],
        "train_loss": training_history['train_loss'][-1],
        "best_epoch_performance": max(training_history['val_f1']) # or however you track best
    }
    
    # Add 'model_name' to model_cfg so it appears in the ID
    current_model_cfg["model_name"] = model_name

    # Save everything
    exp_id = registry.save_experiment(
        model=model,
        optimizer=optimizer,
        train_cfg=current_train_cfg,
        model_cfg=current_model_cfg,
        metrics=final_metrics
    )
    
    # Log the ID to Comet so you can link them
    comet_experiment.log_other("local_experiment_id", exp_id)
    comet_experiment.end()
    
    return model, training_history


def load_model(self, exp_id, model_class, device):
        """Helper to load a model by ID"""
        if exp_id not in self.registry:
            raise ValueError(f"ID {exp_id} not found in registry.")
            
        entry = self.registry[exp_id]
        path = entry["model_path"]
        config = entry["model_architecture"]
        
        # Instantiate
        model = model_class(**config)
        
        # Load Weights
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(device)