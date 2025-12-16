import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionMIL(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = 'resnet18', pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        
        # --- 1. Feature Extractor (Backbone) ---
        # Questo trasformerà ogni tile in un vettore di feature
        self.backbone = models.get_model(backbone_name, weights='IMAGENET1K_V1' if pretrained else None)
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Rimuoviamo il classificatore
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # --- 2. Meccanismo di Attention ---
        # Questo network imparerà a dare un peso a ogni tile
        self.attention = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # --- 3. Classificatore finale ---
        # Questo prenderà il vettore aggregato e farà la previsione per la slide
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x è una lista di "sacchi" (bags). Durante il training, il batch size è > 1.
        # Durante l'inferenza, di solito è 1.
        # Questo loop gestisce un batch di sacchi (es. 4 slide, ognuna con N tile).
        
        slide_outputs = []
        for bag in x: # 'bag' è un tensore di shape [N_tiles, Canali, H, W]
            if bag.size(0) == 0: continue # Salta sacchi vuoti

            # --- Passa tutti i tile nel backbone ---
            # Dobbiamo farlo in mini-batch per non esaurire la memoria GPU se N_tiles è grande
            all_features = []
            for tile_batch in torch.split(bag, 64): # Processa 64 tile alla volta
                features = self.backbone(tile_batch)
                all_features.append(features)
            
            bag_features = torch.cat(all_features, dim=0) # Shape: [N_tiles, num_features]
            
            # --- Calcola i pesi di attenzione ---
            A = self.attention(bag_features) # Shape: [N_tiles, 1]
            A = torch.transpose(A, 1, 0)     # Shape: [1, N_tiles]
            A = F.softmax(A, dim=1)          # I pesi ora sommano a 1
            
            # --- Crea il vettore aggregato della slide ---
            # Media pesata dei feature vector dei tile
            slide_feature_vector = torch.mm(A, bag_features) # Shape: [1, num_features]
            
            # --- Classifica la slide ---
            prediction = self.classifier(slide_feature_vector) # Shape: [1, num_classes]
            slide_outputs.append(prediction)
            
        if not slide_outputs:
            # Se il batch era vuoto, restituisci qualcosa di valido
            return torch.empty(0, self.classifier[-1].out_features).to(x[0].device)

        return torch.cat(slide_outputs, dim=0)