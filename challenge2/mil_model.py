import torch
import torch.nn as nn
import torch.nn.functional as F
# Non abbiamo più bisogno di torchvision.models qui
import timm

class AttentionMIL(nn.Module):
    """
    Modello Multiple Instance Learning con Attention e un backbone flessibile da TIMM.
    """
    def __init__(
        self, 
        num_classes: int, 
        backbone_name: str = 'vit_small_patch16_224',
        pretrained: bool = True,
        freeze_backbone: bool = True, # <-- ORA È UN ARGOMENTO
        embedding_dim: int = 256 # <-- Ridotto a 256 per regolarizzare un po'
    ):
        super().__init__()
        
        # --- 1. Feature Extractor (Backbone) ---
        # Crea il modello da timm. `pretrained=True` carica i pesi di ImageNet-1k.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        
        # Ottieni la dimensione delle feature dal backbone
        # Per i ViT, si chiama 'head.in_features', per le CNN 'fc.in_features'
        if hasattr(self.backbone, 'head'):
            self.num_features = self.backbone.head.in_features
            # Rimpiazza la testa di classificazione con un layer identità
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # Fallback per altre architetture
            # Questo potrebbe richiedere un'ispezione manuale del modello
            raise NotImplementedError(f"Backbone '{backbone_name}' non ha una testa 'head' o 'fc' standard.")

        # --- MODIFICA CORRETTA: Applica il freeze SE richiesto ---
        if freeze_backbone:
            print("--- ATTENZIONE: Backbone è freezato. Solo la testa di classificazione verrà addestrata. ---")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # --- 2. Meccanismo di Attention ---
        # Questo impara un peso (importanza) per ogni tile
        self.attention_net = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # --- 3. Classificatore Finale ---
        # Questo prende il vettore aggregato e fa la previsione per l'intera slide
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # x è una lista di "sacchi" (bags) di tile
        slide_outputs = []
        
        # Itera su ogni slide nel batch
        for bag in x:
            if bag.size(0) == 0: continue

            # Estrai le feature da tutti i tile della slide
            # Lo facciamo in mini-batch per gestire slide con molti tile
            all_features = []
            for tile_batch in torch.split(bag, 64):
                with torch.no_grad() if self.backbone.training is False else torch.enable_grad():
                    features = self.backbone(tile_batch)
                all_features.append(features)
            
            bag_features = torch.cat(all_features, dim=0) # Shape: [N_tiles, num_features]
            
            # Calcola i pesi di attenzione per ogni tile
            A_raw = self.attention_net(bag_features) # Shape: [N_tiles, 1]
            A = torch.transpose(A_raw, 1, 0)         # Shape: [1, N_tiles]
            A = F.softmax(A, dim=1)                  # Pesi normalizzati che sommano a 1
            
            # Crea il vettore aggregato della slide (media pesata delle feature)
            slide_feature_vector = torch.mm(A, bag_features) # Shape: [1, num_features]
            
            # Classifica la slide
            prediction = self.classifier(slide_feature_vector) # Shape: [1, num_classes]
            slide_outputs.append(prediction)
            
        if not slide_outputs:
            return torch.empty(0, self.classifier[-1].out_features).to(x[0].device)

        return torch.cat(slide_outputs, dim=0)