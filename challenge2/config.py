from pathlib import Path

# Set up loss function and optimizer
#criterion = nn.CrossEntropyLoss() --> substituted by the following
MODEL_NAME = "CNN"


# --- config.py ---

# 1. SHARED TRAINING PARAMETERS
# Default values used if you don't override them in start_training()
TRAINING_DEFAULTS = {
    "epochs": 1000,
    "learning_rate": 1e-3,
    "patience": 50,
    "l1_lambda": 0,
    "l2_lambda": 0,
    "verbose": 10,
    "criterion_name" : "CrossEntropyLoss", # possible values: "CrossEntropyLoss"
    "optimizer_name" : "adamw",
}

LOADER_PARAMS = {
    "batch_size": 128,
    "percentage_validation": 0.2
}

# 2. VANILLA CNN CONFIGURATION
CNN_DEFAULTS = {
    "input_shape": (3, 32, 32),
    "num_classes": 4,
    "num_blocks": 2,
    "convs_per_block": 1,
    "use_stride": False,
    "stride_value": 2,
    "padding_size": 1,
    "pool_size": 2,
    "initial_channels": 32,
    "channel_multiplier": 2,
    "dropout_rate_classifier_head": 0.2
}

# 3. EFFICIENTNET CONFIGURATION
EFFICIENTNET_DEFAULTS = {
    "input_shape": (3, 32, 32),
    "num_classes": 4,      # same as "output_shape"
    "filters": 32,
    "kernel_size": 3,
    "stack": 2,            
    "blocks": 2
}

# 4. RESNET CONFIGURATION

RESNET_DEFAULTS = {
    "num_classes": 4,
    "use_pretrained": True,
    "backbone": "resnet18",
    "input_channels": 4  # default expects RGB + mask; override to 3 when training on RGB-only tiles
}


# 5. DIRECTORIES
#BASE_DATA = Path("../../drive/MyDrive/AN2DL_Challenge2-TheBigBatchTheory/data") # path for colab
BASE_DATA = Path("./data") # local path
BASE_DATASET = BASE_DATA / "dataset"
BASE_PREPROCESSED = BASE_DATA / "preprocessed"

TRAIN_DIR = BASE_DATASET / "train_data"
TEST_DIR = BASE_DATASET / "test_data"
LABELS_CSV = BASE_DATASET / "train_labels.csv"

LABEL_MAP = {
    "Luminal A": 0,
    "Luminal B": 1,
    "HER2-(+)": 2,
    "Triple negative": 3
    }