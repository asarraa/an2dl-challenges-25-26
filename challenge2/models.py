import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


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
    def __init__(self, input_shape, num_classes, filters=32, kernel_size=3, stack=2, blocks=3, freeze_backbone=False):
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

        # 2. FREEZING (Congelamento)
        # ---------------------------------------------
        # Lo facciamo PRIMA di definire i layer finali, così siamo sicuri
        # di congelare solo ciò che c'è "prima".
        if freeze_backbone:
            # Congela il layer iniziale
            for param in self.init_conv.parameters():
                param.requires_grad = False
            
            # Congela tutti i blocchi MBConv
            for param in self.blocks_list.parameters():
                param.requires_grad = False
            
            print("🔒 EfficientNet Backbone congelato (pesi fissati).")
        
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
        return x
        #return F.softmax(x, dim=1)


class HistologyResNet(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True, backbone='resnet18', input_channels=4, dropout_rate=0.3, freeze_backbone=False):
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

        # --- FREEZING (NUOVO STEP) ---
        # Lo facciamo SUBITO DOPO aver caricato il modello base, ma PRIMA delle "chirurgie".
        # In questo modo congeliamo tutto ciò che è pre-addestrato.
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Backbone '{backbone}' freezed (fixed weights).")
            
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
            nn.Dropout(p=dropout_rate),
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
    
class PretrainedEfficientNet(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=False, dropout_rate=0.5):
        super().__init__()
        
        # 1. Carica il modello pre-addestrato ufficiale
        # "DEFAULT" scarica i pesi migliori disponibili su ImageNet
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        
        # 2. Freezing (Opzionale ma raccomandato all'inizio)
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
            print("🔒 Backbone EfficientNet congelato.")
            
        # 3. Sostituzione della testa (Classifier)
        # In EfficientNet, il classificatore si chiama 'classifier' ed è un Sequential.
        # L'ultimo layer lineare è classifier[1].
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class GeM(nn.Module):
    """Generalized Mean Pooling used by the fine-tuned ResNet."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        return torch.pow(torch.mean(torch.pow(x, self.p), dim=(-2, -1), keepdim=True), 1.0 / self.p)


class FineTunedResNet50(nn.Module):
    """
    Strong baseline for histology tiles based on a pretrained ResNet50.
    - Supports 3 or 4 channels (mask) by adapting the stem.
    - Optional GeM pooling and partial freezing of early layers.
    """

    def __init__(
        self,
        num_classes=4,
        input_channels=3,
        use_pretrained=True,
        dropout_rate=0.45,
        classifier_hidden=512,
        freeze_backbone=False,
        freeze_until="layer2",
        global_pool="gem",
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Adapt first conv to additional channels (e.g., add_mask_channel=True)
        self._adapt_input_conv(input_channels)

        # Replace pooling/head to allow custom classifier
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if global_pool.lower() == "gem":
            self.backbone.avgpool = GeM()
        else:
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self._freeze_until(freeze_until)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _adapt_input_conv(self, input_channels: int):
        conv1 = self.backbone.conv1
        if input_channels == conv1.in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias,
        )

        with torch.no_grad():
            copy_channels = min(input_channels, conv1.in_channels)
            new_conv.weight[:, :copy_channels, :, :] = conv1.weight[:, :copy_channels, :, :]

            if input_channels > conv1.in_channels:
                extra = input_channels - conv1.in_channels
                mean_weight = conv1.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, conv1.in_channels:input_channels, :, :] = mean_weight.repeat(1, extra, 1, 1)
        self.backbone.conv1 = new_conv

    def _freeze_until(self, freeze_until: str):
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until not in freeze_order:
            return

        for name, module in self.backbone.named_children():
            if name in freeze_order:
                for param in module.parameters():
                    param.requires_grad = False
                if name == freeze_until:
                    break

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class HistologyDenseNet(nn.Module):
    def __init__(self, num_classes=4, input_channels=3, pretrained=True, freeze_backbone=True):
        super(HistologyDenseNet, self).__init__()
        
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)
        
        # --- GESTIONE CANALI DIVERSI DA 3 (Opzionale ma utile) ---
        # Se in futuro userai maschere (4 canali), questo codice adatterà il primo layer
        if input_channels != 3:
            print(f"[INFO] Modifica primo layer DenseNet per {input_channels} canali.")
            original_conv = self.model.features.conv0
            
            # Crea nuova conv con N canali input, ma stessi pesi/parametri
            self.model.features.conv0 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # (Nota: i pesi per i canali extra saranno inizializzati random)

        if freeze_backbone and pretrained:
            self._freeze_all_layers()
            
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    # --- Metodi Helper per il Freezing ---
    
    def _freeze_all_layers(self):
        """Blocca tutta la backbone (Fase 1)"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        print("Backbone DenseNet bloccata.")

    def unfreeze_last_block(self):
        """Sblocca l'ultimo Dense Block per il Fine-Tuning (Fase 2)"""
        # Sblocca la Norm finale e l'ultimo blocco denso
        for name, param in self.model.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False # Sicurezza
        
        # Assicuriamoci che il classificatore sia sempre sbloccato
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
        print("Sbloccato denseblock4, norm5 e classifier.")
        
class FineTunedResNet18(nn.Module):
    """
    Lightweight baseline for histology tiles based on a pretrained ResNet18.
    - Less prone to overfitting on small/noisy datasets compared to ResNet50.
    - Supports 3 or 4 channels (mask).
    - Optional GeM pooling and partial freezing.
    """

    def __init__(
        self,
        num_classes=4,
        input_channels=3,
        use_pretrained=True,
        dropout_rate=0.5,
        classifier_hidden=256,
        freeze_backbone=False,
        freeze_until="layer1",
        global_pool="gem",
    ):
        super().__init__()

        # Carica pesi ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Adatta il primo conv per canali addizionali (es. maschera)
        self._adapt_input_conv(input_channels)

        # Sostituisci pooling e fully connected layer
        self.in_features = self.backbone.fc.in_features # 512 per ResNet18
        self.backbone.fc = nn.Identity()
        
        if global_pool.lower() == "gem":
            self.backbone.avgpool = GeM()
        else:
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Logica di congelamento
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self._freeze_until(freeze_until)

        # Classificatore custom
        self.classifier = nn.Sequential(
            nn.Flatten(), # Assicura che l'input sia piatto
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25), # Secondo dropout mantenuto
            nn.Linear(classifier_hidden, num_classes),
        )

    def _adapt_input_conv(self, input_channels: int):
        conv1 = self.backbone.conv1
        if input_channels == conv1.in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias,
        )

        with torch.no_grad():
            # Copia i pesi esistenti per i primi canali
            copy_channels = min(input_channels, conv1.in_channels)
            new_conv.weight[:, :copy_channels, :, :] = conv1.weight[:, :copy_channels, :, :]

            # Inizializza canali extra con la media dei pesi RGB
            if input_channels > conv1.in_channels:
                extra = input_channels - conv1.in_channels
                mean_weight = conv1.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, conv1.in_channels:input_channels, :, :] = mean_weight.repeat(1, extra, 1, 1)
        
        self.backbone.conv1 = new_conv

    def _freeze_until(self, freeze_until: str):
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until not in freeze_order:
            return

        # Itera sui figli per congelare in ordine
        reached_target = False
        for name, module in self.backbone.named_children():
            if reached_target:
                break 
            
            if name in freeze_order:
                for param in module.parameters():
                    param.requires_grad = False
                if name == freeze_until:
                    reached_target = True

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    



class DualBranchResNet(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = 'resnet18', pretrained: bool = True, freeze_backbones: bool = False, dropout_rate: float = 0.5):
        super().__init__()
        
        # --- Ramo 1: Contesto ---
        self.context_backbone = models.get_model(backbone_name, weights='IMAGENET1K_V1' if pretrained else None)
        context_features_in = self.context_backbone.fc.in_features
        self.context_backbone.fc = nn.Identity()

        # --- Ramo 2: Dettaglio ---
        self.detail_backbone = models.get_model(backbone_name, weights='IMAGENET1K_V1' if pretrained else None)
        detail_features_in = self.detail_backbone.fc.in_features
        self.detail_backbone.fc = nn.Identity()
        
        if freeze_backbones:
            for param in self.context_backbone.parameters():
                param.requires_grad = False
            for param in self.detail_backbone.parameters():
                param.requires_grad = False

        # --- Testa di Classificazione ("Classifier Head") ---
        # Questa è la struttura standard e più robusta.
        
        combined_features_dim = context_features_in + detail_features_in
        
        self.classifier = nn.Sequential(
            # 1. Primo blocco: riduce la dimensionalità e regolarizza
            nn.Linear(combined_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout DOPO l'attivazione

            # 2. Layer finale di output (Logits)
            # Di solito non si applica dropout o attivazione direttamente prima dell'output
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_context, x_detail = x
        
        # Estrai le feature. Non applichiamo dropout qui.
        features_context = self.context_backbone(x_context)
        features_detail = self.detail_backbone(x_detail)
        
        # Concatena le feature
        combined_features = torch.cat([features_context, features_detail], dim=1)
        
        # Passa le feature combinate e pulite al classificatore
        output = self.classifier(combined_features)
        
        return output

class FineTunedVGG16(nn.Module):
    """
    Versione adattata di VGG16_BN per istologia.
    - Sostituisce il flatten layer originale con Global Average Pooling (o GeM)
      per ridurre drasticamente i parametri e l'overfitting.
    - Supporta N canali in input.
    - Gestisce il freezing per blocchi.
    """

    def __init__(
        self,
        num_classes=4,
        input_channels=3,
        use_pretrained=True,
        dropout_rate=0.5,
        classifier_hidden=512,
        freeze_backbone=False,
        freeze_until="block2", # block1, block2, block3, block4
        global_pool="gem",
    ):
        super().__init__()

        # 1. Carica VGG16 con Batch Normalization (Fondamentale per la convergenza)
        weights = models.VGG16_BN_Weights.DEFAULT if use_pretrained else None
        original_vgg = models.vgg16_bn(weights=weights)
        
        # VGG separa le feature (conv) dal classificatore. Teniamo solo le feature.
        self.features = original_vgg.features

        # 2. Adatta il primo layer per canali extra (es. maschera)
        self._adapt_input_conv(input_channels)

        # 3. Pooling Strategy
        # VGG esce con 512 canali. Usiamo pooling per ottenere (B, 512, 1, 1)
        self.in_features = 512
        if global_pool.lower() == "gem":
            self.avgpool = GeM()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Freezing
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            self._freeze_until(freeze_until)

        # 5. Nuovo Classificatore Custom (Molto più leggero dell'originale VGG)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.in_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2), # Un po' meno dropout nel secondo step
            nn.Linear(classifier_hidden, num_classes),
        )

    def _adapt_input_conv(self, input_channels: int):
        # In VGG16, il primo layer è features[0]
        conv1 = self.features[0]
        
        if input_channels == conv1.in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )

        with torch.no_grad():
            # Copia i pesi esistenti
            copy_channels = min(input_channels, conv1.in_channels)
            new_conv.weight[:, :copy_channels, :, :] = conv1.weight[:, :copy_channels, :, :]

            # Se ci sono nuovi canali, inizializzali con la media dei canali RGB
            if input_channels > conv1.in_channels:
                extra = input_channels - conv1.in_channels
                mean_weight = conv1.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, conv1.in_channels:input_channels, :, :] = mean_weight.repeat(1, extra, 1, 1)
            
            # Copia il bias se esiste
            if conv1.bias is not None:
                new_conv.bias = conv1.bias

        self.features[0] = new_conv

    def _freeze_until(self, freeze_until: str):
        """
        Congela i layer fino al blocco specificato.
        VGG è sequenziale, quindi mappiamo i nomi agli indici della lista 'features'.
        Gli indici corrispondono ai MaxPool2d che chiudono i blocchi in VGG16_BN.
        """
        # Mappa approssimativa dei blocchi in VGG16_BN
        # block1 finisce a index 6
        # block2 finisce a index 13
        # block3 finisce a index 23
        # block4 finisce a index 33
        # block5 finisce a index 43
        block_indices = {
            "block1": 6,
            "block2": 13,
            "block3": 23,
            "block4": 33,
            "block5": 43
        }

        if freeze_until not in block_indices:
            return

        stop_index = block_indices[freeze_until]
        
        # Itera su tutti i layer sequenziali
        for i, layer in enumerate(self.features):
            for param in layer.parameters():
                param.requires_grad = False
            
            # Se abbiamo raggiunto la fine del blocco richiesto, fermati
            if i >= stop_index:
                break

    def forward(self, x):
        x = self.features(x)  # Estrazione feature (B, 512, H/32, W/32)
        x = self.avgpool(x)   # Pooling (B, 512, 1, 1)
        x = self.classifier(x) # Classificazione
        return x
