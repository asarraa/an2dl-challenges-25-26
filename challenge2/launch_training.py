import os
import torch
import torch.nn as nn
import mil_pipeline
import multiscale_pipeline
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from comet_ml import start
import mil_model

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
        # Remove parameters that DualBranchResNet doesn't accept
        invalid_params = ['input_shape', 'use_pretrained', 'input_channels']
        for param in invalid_params:
            if param in cfg_copy:
                del cfg_copy[param]
        # Map 'use_pretrained' to 'pretrained' if needed
        if 'pretrained' not in cfg_copy:
            cfg_copy['pretrained'] = True
        model = models.DualBranchResNet(**cfg_copy)
    elif model_name == "AttentionMIL":
        cfg_copy = current_model_cfg.copy()
        model = mil_model.AttentionMIL_UNI(**cfg_copy)
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

# Import necessari all'inizio del tuo file launch_training.py
import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import login
from torch.utils.tensorboard import SummaryWriter # Assicurati di importare SummaryWriter
from comet_ml import Experiment # Assumendo che usi start da un altro file

# Importa i tuoi moduli personalizzati
import mil_model
import mil_pipeline
from training_engine import fit

# =============================================================================
# --- FUNZIONE PRINCIPALE DI ORCHESTRAZIONE DEL TRAINING (Versione MIL) ---
# =============================================================================

def start_training(
    model_name: str, 
    training_params: dict,
    data_path: Path,
    batch_size: int,
    device: torch.device,
    comet_experiment: Experiment, # Passa l'esperimento Comet
    local_data_path: Path,
    debug_mode: bool = False,
    pretrained_model_path: Path = None # Per riprendere un training
):
    """
    Orchestra l'intero processo di training per il modello MIL con backbone UNI.
    """
    print(f"--- Avvio del processo di training per il modello: {model_name} ---")
    
    # --- FASE 0: Autenticazione a Hugging Face ---
    try:
        login()
    except Exception as e:
        print(f"ATTENZIONE: Login a Hugging Face fallito o già eseguito.")

    # --- FASE 1: Ottenimento delle trasformazioni corrette dal modello UNI ---
    print("\n--- [FASE 1] Ottenimento delle trasformazioni specifiche di UNI ---")
    try:
        # Crea un modello fittizio solo per accedere alla sua configurazione
        dummy_model = mil_model.AttentionMIL(num_classes=training_params['num_classes'])
        uni_transforms = mil_model.get_uni_transforms(dummy_model.backbone)
        del dummy_model
        print("✅ Trasformazioni per UNI ottenute con successo.")
    except Exception as e:
        print(f"❌ ERRORE CRITICO: Impossibile ottenere le trasformazioni di UNI. {e}")
        return None, None, None

    # --- FASE 2: Creazione dei DataLoaders ---
    print("\n--- [FASE 2] Creazione dei DataLoaders Multi-Istanza ---")
    try:
        train_loader, val_loader, _, class_weights = mil_pipeline.get_mil_loaders(
            base_path=data_path,
            batch_size=batch_size,
            uni_transform=uni_transforms
        )
        print("✅ DataLoaders creati con successo.")
    except Exception as e:
        print(f"❌ ERRORE CRITICO: Creazione DataLoaders fallita. {e}")
        return None, None, None

    # --- FASE 3: Istanziazione del modello ---
    print("\n--- [FASE 3] Istanziazione del modello per il training ---")
    try:
        if model_name != "AttentionMIL_UNI":
            raise ValueError(f"Questo script è configurato solo per 'AttentionMIL_UNI', non '{model_name}'.")
        
        model = mil_model.AttentionMIL(
            num_classes=training_params['num_classes'],
            backbone_name=training_params.get('backbone_name', 'hf-hub:MahmoodLab/UNI2-h'),
            freeze_backbone=training_params.get('freeze_backbone', True),
            dropout_rate=training_params.get('dropout_rate', 0.5)
        )
        
        if pretrained_model_path:
            print(f"Caricamento pesi da un training precedente: {pretrained_model_path}...", flush=True)
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print("Pesi caricati con successo.", flush=True)

        model.to(device)
        print(f"✅ Modello '{model_name}' istanziato e spostato su '{device}'.")
        print(f"   - Backbone freezato: {training_params.get('freeze_backbone', True)}")
        
    except Exception as e:
        print(f"❌ ERRORE CRITICO: Istanziazione modello fallita. {e}")
        return None, None, None

    # --- FASE 4: Definizione degli oggetti di training ---
    print("\n--- [FASE 4] Configurazione degli oggetti di training ---")
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Logica per i learning rate differenziati
    backbone_lr = training_params.get('backbone_lr', training_params['learning_rate'])
    classifier_lr = training_params.get('classifier_lr', training_params['learning_rate'])
    
    optimizer_params = [
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.attention_net.parameters(), 'lr': classifier_lr},
        {'params': model.classifier.parameters(), 'lr': classifier_lr}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=training_params['l2_lambda'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_params['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    print(f"   - Criterion: CrossEntropyLoss (con class weights)")
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning Rates: Backbone={backbone_lr}, Head={classifier_lr}")
    print(f"   - Scheduler: CosineAnnealingLR")

    # --- FASE 5: Avvio del training ---
    print("\n--- [FASE 5] Avvio del ciclo di training (fit) ---")
    
    run_id = comet_experiment.id if comet_experiment else "local_run"
    experiment_name = f"{model_name}_{run_id[:8]}"
    
    # TensorBoard (opzionale)
    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    model, training_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_params['epochs'],
        criterion=criterion,
        optimizer=optimizer, 
        scheduler=scheduler,
        scaler=scaler,
        device=device, 
        writer=writer,
        patience=training_params['patience'],
        experiment_name=experiment_name, 
        comet_experiment=comet_experiment,
        debug_mode=debug_mode,
        local_data_path=local_data_path
    )

    
    comet_experiment.end()
    
    return model, training_history, run_id