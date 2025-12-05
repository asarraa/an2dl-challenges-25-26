# -----------------------------
# Custom library imports
# -----------------------------
import config
import models
import preprocessing
from comet_ml import start #for logging
from training_engine import train_one_epoch, validate_one_epoch, fit, log_metrics_to_tensorboard

# -----------------------------
# Import libraries
# -----------------------------

# Set seed for reproducibility
SEED = 42

# Import necessary libraries
import os

# Set environment variables before importing modules
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import logging
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(SEED)
random.seed(SEED)

# Import PyTorch
import torch
torch.manual_seed(SEED)
from torch import nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Configurazione di TensorBoard e directory
logs_dir = "tensorboard"
#TODO: controlla come lanciare tensorboard
'''
!pkill -f tensorboard
%load_ext tensorboard
!mkdir -p models'''

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

# Import other libraries
import copy
import shutil
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.gridspec as gridspec

# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)




# -----------------------------
# Custom functions
# -----------------------------

def initialize_training():
    # Initialize best model tracking variables
    best_model = None
    best_performance = float('-inf')

def instantiate_model(name, batch_size):
    if name == "CNN":
        # Instantiate CNN model and move to computing device (CPU/GPU)
        model = models.CNN(
            config.input_shape,
            config.num_classes,
            num_blocks=config.NUM_BLOCKS,
            convs_per_block=config.CONVS_PER_BLOCK,
            use_stride=config.USE_STRIDE,
            stride_value=config.STRIDE_VALUE,
            padding_size=config.PADDING_SIZE,
            pool_size=config.POOL_SIZE,
            initial_channels=config.INITIAL_CHANNELS,
            channel_multiplier=config.CHANNEL_MULTIPLIER
            ).to(device)

    elif name == "EfficientNet":
        # Create and display the DenseNet model
        model = models.EfficientNetModel(config.input_shape, config.output_shape, config.filters, config.kernel_size, config.stack, config.blocks).to(device)


    summary(model, input_size=config.input_shape)
    model_graph = draw_graph(model, input_size=(batch_size)+config.input_shape, expand_nested=True, depth=5)
    model_graph.visual_graph


def get_criterion_from_name(name):
    if config.LOSS_FN == name:
        criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer_and_scaler_from_name(name, model, learning_rate, l2_lambda):
    # Define optimizer with L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Enable mixed precision training for GPU acceleration
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    return optimizer, scaler


def start_training(model_name=config.MODEL_NAME, train_loader=preprocessing.train_loader, val_loader=preprocessing.val_loader, epochs=config.epochs, criterion_name=config.CRITERION_NAME, optimizer_name=config.OPTIMIZER_NAME, scaler=config.scaler, device=device, writer=config.writer, learning_rate = config.LEARNING_RATE, l1_lambda = config.L1_LAMBDA ,l2_lambda=config.L2_LAMBDA, verbose=config.VERBOSE, experiment_name="", patience=config.PATIENCE, batch_size = config.BATCH_SIZE):
#    %%time   Do we need it?

    initialize_training
    model = instantiate_model(model_name, batch_size)
    criterion = get_criterion_from_name(criterion_name)
    optimizer, scaler = get_optimizer_and_scaler_from_name(optimizer_name, learning_rate, l2_lambda)
    experiment_name = model_name + "parameters:" #TODO

        #Initialize Comet logging
    comet_experiment = start(
      api_key="nhvfD4vUpZNMoJQ3dEjOwIeua",
      project_name="test",
      workspace="asarraa"
    )

    hyper_params = {
      "learning_rate": learning_rate,
      "batch_size": batch_size,
      "epochs": epochs,
      "model": model,
    }


    # Train model and track training history
    model, training_history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer, 
        scaler=scaler,
        device=device, 
        writer=writer,
        l1_lambda=l1_lambda,
        l2_lambda=0,
        verbose=verbose,
        experiment_name=experiment_name, #TODO:check names
        patience=patience,
        comet_experiment=comet_experiment
        )

    # Update best model if current performance is superior
    if training_history['val_f1'][-1] > best_performance:
        best_model = model
        best_performance = training_history['val_f1'][-1]

    comet_experiment.log_parameters(hyper_params)