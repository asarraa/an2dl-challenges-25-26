import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import torchvision.models as models


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
    def __init__(self, input_shape=(3,32,32), num_classes=10,
                 num_blocks=2, convs_per_block=1,
                 use_stride=False, stride_value=2, padding_size=1, pool_size=2,
                 initial_channels=32, channel_multiplier=2, dropout_rate_classifier_head=0.2):
        super().__init__()
        print("[DEBUG] Initializing CNN model with the following parameters:")
        print(f"input_shape: {input_shape}")
        print(f"num_classes: {num_classes}")
        print(f"num_blocks: {num_blocks}")
        print(f"convs_per_block: {convs_per_block}")
        print(f"use_stride: {use_stride}")
        print(f"stride_value: {stride_value}")
        print(f"padding_size: {padding_size}")
        print(f"pool_size: {pool_size}")
        print(f"initial_channels: {initial_channels}")
        print(f"channel_multiplier: {channel_multiplier}")
        print(f"dropout_rate_classifier_head: {dropout_rate_classifier_head}")

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
            nn.Dropout(dropout_rate_classifier_head),
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
    def __init__(self, input_shape, num_classes, filters=32, kernel_size=3, stack=2, blocks=3):
        """Initialises the EfficientNetModel.

        Args:
            input_shape (tuple): Shape of the input images (C, H, W).
            num_classes (int): Number of output classes.
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
        self.dense = nn.Linear(current_channels, num_classes) # Final fully connected layer

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


class HistologyResNet(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True, backbone='resnet18', input_channels=4):
        """
        Modello basato su ResNet con supporto a 3 o 4 canali.
        Usa i pesi pre-addestrati di ImageNet per i canali RGB e inizializza
        eventuali canali extra replicando la media dei pesi RGB.
        """
        super().__init__()
        
        # 1. Carichiamo la backbone pre-addestrata
        # ResNet18 è leggera e veloce. Se hai molta GPU usa 'resnet50'
        if backbone == 'resnet18':
            self.model = models.resnet18(weights='DEFAULT' if use_pretrained else None)
            last_channel_in = self.model.fc.in_features
        elif backbone == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT' if use_pretrained else None)
            last_channel_in = self.model.fc.in_features
        else:
            raise ValueError("Backbone supportata: resnet18, resnet50")

        # ---------------------------------------------------------
        # 2. CHIRURGIA DEL PRIMO LIVELLO (Input Layer Surgery)
        # ---------------------------------------------------------
        original_conv1 = self.model.conv1

        # Se il numero di canali coincide con l'originale (3), non serve cambiare nulla.
        if input_channels != original_conv1.in_channels:
            new_conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )

            if use_pretrained:
                with torch.no_grad():
                    # Copia i primi canali disponibili (fino a 3)
                    copy_channels = min(input_channels, original_conv1.in_channels)
                    new_conv1.weight[:, :copy_channels, :, :] = original_conv1.weight[:, :copy_channels, :, :]

                    # Se servono canali extra (es. maschera), inizializza con la media dei pesi RGB
                    if input_channels > original_conv1.in_channels:
                        extra_channels = input_channels - original_conv1.in_channels
                        mean_weight = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                        new_conv1.weight[:, original_conv1.in_channels:input_channels, :, :] = mean_weight.repeat(1, extra_channels, 1, 1)

            self.model.conv1 = new_conv1
        
        # ---------------------------------------------------------
        # 3. CHIRURGIA DELLA TESTA (Classification Head)
        # ---------------------------------------------------------
        # Sostituiamo l'ultimo layer fully connected per matchare le nostre classi
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3), # Aggiungiamo un po' di dropout per evitare overfitting
            nn.Linear(last_channel_in, num_classes)
        )

    def forward(self, x):
        return self.model(x)



class VanillaCNNBlockCustom(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1, use_stride=False, stride_value=2, padding_size=1, pool_size=2):
        super().__init__()
        layers = []
        
        # 1. Gestiamo le convoluzioni multiple
        for i in range(num_convs):
            # Logica per gestire input/output channels
            cin = in_channels if i == 0 else out_channels
            cout = out_channels
            
            # Logica per lo stride (lo applichiamo solo all'ultima conv se richiesto)
            # Nota: applicare stride all'ultima conv è tipico di ResNet, 
            # applicare MaxPool alla fine è tipico di VGG.
            current_stride = 1
            if use_stride and i == (num_convs - 1):
                current_stride = stride_value
            
            layers.append(nn.Conv2d(cin, cout, kernel_size=3, padding=padding_size, stride=current_stride))
            layers.append(nn.BatchNorm2d(cout)) # <--- Aggiunta Cruciale
            layers.append(nn.ReLU())            # <--- Aggiunta Cruciale dopo OGNI conv

        # 2. Pooling (se non usiamo stride per downsampling)
        if not use_stride:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

    # Convolutional Neural Network architecture for CIFAR10 classification
#N.B. If use_stride is true, we apply downsapling with stride, otherwise we downsample with pooling
class CNNCustom(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10,
                 num_blocks=2, convs_per_block=1,
                 use_stride=False, stride_value=2, padding_size=1, pool_size=2,
                 initial_channels=32, channel_multiplier=2, dropout_rate_classifier_head=0.2):
        super().__init__()
        print("[DEBUG] Initializing CNN model with the following parameters:")
        print(f"input_shape: {input_shape}")
        print(f"num_classes: {num_classes}")
        print(f"num_blocks: {num_blocks}")
        print(f"convs_per_block: {convs_per_block}")
        print(f"use_stride: {use_stride}")
        print(f"stride_value: {stride_value}")
        print(f"padding_size: {padding_size}")
        print(f"pool_size: {pool_size}")
        print(f"initial_channels: {initial_channels}")
        print(f"channel_multiplier: {channel_multiplier}")
        print(f"dropout_rate_classifier_head: {dropout_rate_classifier_head}")

        # Build convolutional blocks
        blocks = []
        in_channels = input_shape[0]
        out_channels = initial_channels

        #append single CNN Blocks defined in the VanillaCNNBlock class
        for i in range(num_blocks):
            blocks.append(VanillaCNNBlockCustom(
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
            nn.Dropout(dropout_rate_classifier_head),
            nn.Linear(flattened_size, num_classes)
        )

    # Forward pass through the network
    def forward(self, x):
        x = self.features(x)
        x = self.classifier_head(x)
        return x
    




    import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Un blocco che impara la differenza (residuo) invece della funzione completa.
    Molto più facile da addestrare per reti profonde.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Se le dimensioni cambiano (stride=2), adattiamo l'identità
        if self.downsample is not None:
            identity = self.downsample(x)

        # IL SEGRETO: Sommiamo l'input originale all'output (Skip Connection)
        out += identity 
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=4, layers=[2, 2, 2, 2], start_channels=64):
        """
        layers=[2, 2, 2, 2] crea una struttura simile a ResNet18.
        """
        super().__init__()
        self.in_channels = start_channels
        
        # 1. Stem (Ingresso iniziale)
        # Usiamo kernel 7x7 e stride 2 per ridurre subito l'immagine da 224 a 112
        self.conv1 = nn.Conv2d(input_shape[0], start_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(start_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Riduce da 112 a 56

        # 2. Creazione dei Layer (Blocchi Residuali)
        self.layer1 = self._make_layer(start_channels, layers[0])
        self.layer2 = self._make_layer(start_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(start_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(start_channels * 8, layers[3], stride=2)

        # 3. Testa di classificazione con Global Average Pooling (GAP)
        # GAP risolve il problema della RAM e dell'overfitting sui layer lineari
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Dropout alto per regolarizzare
            nn.Linear(start_channels * 8, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        # Se cambiamo dimensioni (stride != 1) o numero di canali, dobbiamo adattare l'identità
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # Il primo blocco gestisce il cambio di dimensione/canali
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # I blocchi successivi mantengono la stessa dimensione
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = self.fc(x)
        return x