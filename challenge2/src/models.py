import config
from torch import nn


# -----------------------------
# Simple CNN (from ex 6)
# -----------------------------

# Single convolutional block with multiple conv layers, ReLU and pooling/stride
class VanillaCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1, use_stride=False, stride_value=2, padding_size=1, pool_size=2):
        super().__init__()

        layers = []

        # First convolution: in_channels -> out_channels
        if num_convs == 1:
            # Single conv: apply stride here if use_stride is True
            stride = stride_value if use_stride else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding_size, stride=stride))
        else:
            # Multiple convs: first one always has stride=1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))

            # Intermediate convolutions (all with stride=1)
            for i in range(1, num_convs - 1):
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))

            # Last convolution: apply stride here if use_stride is True
            stride = stride_value if use_stride else 1
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding_size, stride=stride))

        # ReLU activation
        layers.append(nn.ReLU())

        # Pooling only if not using stride for spatial reduction
        if not use_stride:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Convolutional Neural Network architecture for CIFAR10 classification
#N.B. If use_stride is true, we apply downsapling with stride, otherwise we downsample with pooling
class CNN(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10, dropout_rate=0.2,
                 num_blocks=2, convs_per_block=1,
                 use_stride=False stride_value=2, padding_size=1, pool_size=2,
                 initial_channels=32, channel_multiplier=2):
        super().__init__()

        # Build convolutional blocks
        blocks = []
        in_channels = input_shape[0]
        out_channels = initial_channels

        #append single CNN Blocks defined in the VanillaCNNBlock class
        for i in range(num_blocks):
            blocks.append(VanillaCNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                num_convs=convs_per_block,
                use_stride=use_stride,
                stride_value=stride_value,
                padding_size=padding_size,
                pool_size=pool_size
            ))

            # Prepare for next block: increase channels
            in_channels = out_channels
            out_channels = out_channels * channel_multiplier

        self.features = nn.Sequential(*blocks) #create a sequential layer with the blocks (this is the sequence extractor)

        # Calculate flattened size after all blocks using a dummy forward pass
        # This approach is robust and works with any configuration of padding, stride, and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        # Classification head: flatten features and apply dropout before final layer
        # simple 1 layer feed forward neural network (this is the classification head network)
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, num_classes)
        )

    # Forward pass through the network
    def forward(self, x):
        x = self.features(x)
        x = self.classifier_head(x)
        return x
    



# -----------------------------
# Compound Scaling (EfficientNet), from "Advancements_in_Conv....ipynb"
# -----------------------------

class MBConvBlock(nn.Module):
    """
    MBConv: Expand (1x1) -> Depthwise (3x3) -> SE -> Project (1x1).
    Followed by MaxPool at the end of the stack sequence.
    """
    def __init__(self, in_channels, filters, kernel_size=3, stack=2, expansion=4):
        super().__init__()

        self.units = nn.ModuleList()
        current_in = in_channels

        for s in range(stack):
            unit = nn.ModuleList()
            expanded = current_in * expansion

            # Expansion Phase (1x1 convolution to expand channels)
            if expansion != 1:
                unit.append(nn.Sequential(
                    nn.Conv2d(current_in, expanded, 1, bias=False),
                    nn.BatchNorm2d(expanded),
                    nn.SiLU() # Swish activation
                ))

            # Depthwise Convolution (applies a single filter per input channel)
            unit.append(nn.Sequential(
                nn.Conv2d(expanded, expanded, kernel_size, padding='same', groups=expanded, bias=False),
                nn.BatchNorm2d(expanded),
                nn.SiLU()
            ))

            # Squeeze and Excitation block
            se_in = expanded
            se_reduced = max(1, int(se_in * 0.25))
            unit.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(se_in, se_reduced, 1),
                nn.SiLU(),
                nn.Conv2d(se_reduced, se_in, 1),
                nn.Sigmoid()
            ))

            # Output Projection Phase (1x1 convolution to project channels back)
            unit.append(nn.Sequential(
                nn.Conv2d(expanded, filters, 1, bias=False),
                nn.BatchNorm2d(filters)
            ))

            self.units.append(unit)

            # Update current_in for the next stacked unit
            current_in = filters

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        for unit in self.units:
            residual = x

            # Expand (if expansion factor is not 1)
            out = unit[0](x) if len(unit) == 4 else x
            # Depthwise (index shifts if expansion is skipped)
            dw_idx = 1 if len(unit) == 4 else 0
            out = unit[dw_idx](out)

            # Squeeze and Excitation
            se_w = unit[dw_idx+1](out)
            out = out * se_w

            # Project
            out = unit[dw_idx+2](out)

            # Add residual connection if input and output dimensions match
            if x.shape == out.shape:
                out += x

            x = out

        return self.pool(x)

class EfficientNetModel(nn.Module):
    """Complete CNN model using multiple MBConvBlocks and Global Average Pooling.

    This model integrates the EfficientNet architecture for classification tasks.
    """
    def __init__(self, input_shape, output_shape, filters=32, kernel_size=3, stack=2, blocks=3):
        """Initialises the EfficientNetModel.

        Args:
            input_shape (tuple): Shape of the input images (C, H, W).
            output_shape (int): Number of output classes.
            filters (int, optional): Initial number of filters for the first block. Defaults to 32.
            kernel_size (int, optional): Kernel size for convolutional layers. Defaults to 3.
            stack (int, optional): Number of MBConv units per block. Defaults to 2.
            blocks (int, optional): Number of `MBConvBlock` instances to stack. Defaults to 3.
        """
        super().__init__()

        self.blocks_list = nn.ModuleList()
        current_channels = input_shape[0]
        current_filters = filters

        # Initial Convolutional layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(current_channels, filters, 3, padding='same', bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU()
        )
        current_channels = filters

        # Stack multiple MBConvBlocks, typically doubling filters for each subsequent block
        for b in range(blocks):
            self.blocks_list.append(
                MBConvBlock(current_channels, current_filters, kernel_size, stack)
            )
            current_channels = current_filters
            current_filters *= 2

        self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.flatten = nn.Flatten() # Flatten multi-dimensional output
        self.dense = nn.Linear(current_channels, output_shape) # Final fully connected layer

    def forward(self, x):
        """Defines the forward pass of the EfficientNetModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output probabilities after Softmax activation.
        """
        x = self.init_conv(x)
        for block in self.blocks_list:
            x = block(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)


