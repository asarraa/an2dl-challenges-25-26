
# -----------------------------
# Network training parameters
# -----------------------------
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 1000
PATIENCE = 50

# Regularisation
DROPOUT_RATE = 0.2         # Dropout probability
L1_LAMBDA = 0            # L1 penalty
L2_LAMBDA = 0            # L2 penalty

# Set up loss function and optimizer
#criterion = nn.CrossEntropyLoss() --> substituted by the following
CRITERION_NAME = "crossentropy"

# Print the defined parameters
'''print("Epochs:", EPOCHS)
print("Batch Size:", BATCH_SIZE)
print("Learning Rare:", LEARNING_RATE)
print("Dropout Rate:", DROPOUT_RATE)
print("L1 Penalty:", L1_LAMBDA)
print("L2 Penalty:", L2_LAMBDA)
'''

# -----------------------------
# CNN Architecture parameters
# -----------------------------

INPUT_SHAPE = (3,32,32) #related to data

DROPOUT_RATE_CLASSIFIER_HEAD = DROPOUT_RATE

# Number of convolutional blocks
NUM_BLOCKS = 2

# Number of conv layers per block
CONVS_PER_BLOCK = 1

# Use strided convolutions instead of pooling
USE_STRIDE = False

# Stride value when USE_STRIDE is True
STRIDE_VALUE = 2

# Padding size
PADDING_SIZE = 1

# Pooling size when USE_STRIDE is False
POOL_SIZE = 2

# Number of channels in first block
INITIAL_CHANNELS = 32

# Channel multiplication factor between blocks
CHANNEL_MULTIPLIER = 2

'''print("Num Blocks:", NUM_BLOCKS)
print("Convs per Block:", CONVS_PER_BLOCK)
print("Use Stride:", USE_STRIDE)
print("Stride Value:", STRIDE_VALUE)
print("Padding Size:", PADDING_SIZE)
print("Pool Size:", POOL_SIZE)
print("Initial Channels:", INITIAL_CHANNELS)
print("Channel Multiplier:", CHANNEL_MULTIPLIER)'''

#For efficientNet

# Initialize configuration for convolutional layers
stack = 2
blocks = 2
filters = 32
kernel_size = 3
output_shape = 10


#----
MODEL_NAME = "CNN"
OPTIMIZER_NAME = "crossentropy"


# -----------------------------
# Output parameters
# -----------------------------
VERBOSE = 1