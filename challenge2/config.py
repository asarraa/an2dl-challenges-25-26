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
    "input_shape": (3, 224, 224),
    "num_classes": 4,      # same as "output_shape"
    "filters": 32,
    "kernel_size": 3,
    "stack": 2,            
    "blocks": 4,
    "freeze_backbone": False
}

# 4. RESNET CONFIGURATION

RESNET_DEFAULTS = {
    "num_classes": 4,
    "use_pretrained": True,
    "backbone": "resnet18",
    "input_channels": 3,  # default expects RGB + mask; override to 3 when training on RGB-only tiles
    "dropout_rate": 0.5,
    "freeze_backbone": True
}

PRETRAINED_EFFICIENTNET_DEFAULTS = {
    "input_shape": (3, 224, 224),
    "num_classes": 4,
    "freeze_backbone": True,
    "dropout_rate": 0.5
}

# 5. DIRECTORIES
BASE_FOLDER = Path("./") # path for colab
BASE_DATA = BASE_FOLDER / "data"
BASE_DATASET = BASE_DATA / "dataset"
BASE_PREPROCESSED = BASE_DATA / "preprocessed"

TRAIN_DIR = BASE_DATASET / "train_data"
TEST_DIR = BASE_DATASET / "test_data"
LABELS_CSV = BASE_DATASET / "train_labels.csv"

EXPERIMENTS_DIR = BASE_FOLDER / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"

LABEL_MAP = {
    "Luminal A": 0,
    "Luminal B": 1,
    "HER2(+)": 2,
    "Triple negative": 3
    }




'''
    experiments
        - models
            -- CNN_2025...
            -- ...
        - registry.json

    data --> BASE_DATA
        -dataset --> base_dataset
            --train_data
            --test_data
            --train_labels.csv
            
        -preprocessed --> base_preprocessed 
            --<Preprocess_name.zip> --> single_preprocessing_dir
                [---train
                    ---images
                    ---.csv
                ---test
                    ---images
                    ---.csv]
'''