import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# Non è più necessario il login qui, lo farai nel notebook principale
# from huggingface_hub import login

class AttentionMIL_UNI(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        freeze_backbone: bool = True,
        embedding_dim: int = 512, # Dimensione intermedia del classificatore
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # --- 1. Carica il Backbone UNI2-h ---
        print("--- Loading UNI2-h backbone from Hugging Face Hub... ---")
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, # Cruciale: lo usiamo come feature extractor
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.backbone = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        
        # Ottieni la dimensione delle feature (per UNI2-h è 1536)
        self.num_features = 1536
        
        if freeze_backbone:
            print("--- ATTENZIONE: Backbone UNI2-h è freezato. ---")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # --- 2. Meccanismo di Attention ---
        self.attention_net = nn.Sequential(
            nn.Linear(self.num_features, 256), # Adattato alla nuova dimensione delle feature
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # --- 3. Classificatore Finale ---
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # La logica del forward rimane identica alla versione precedente del MIL
        slide_outputs = []
        for bag in x:
            if bag.size(0) == 0: continue
            
            all_features = []
            # Disattiva i gradienti per il backbone se è freezato, per risparmiare memoria
            with torch.set_grad_enabled(not self.backbone.training is False or any(p.requires_grad for p in self.backbone.parameters())):
                for tile_batch in torch.split(bag, 32): # Batch size più piccolo, il modello è grande
                    features = self.backbone(tile_batch)
                    all_features.append(features)
            
            bag_features = torch.cat(all_features, dim=0)
            
            A_raw = self.attention_net(bag_features)
            A = F.softmax(torch.transpose(A_raw, 1, 0), dim=1)
            
            slide_feature_vector = torch.mm(A, bag_features)
            
            prediction = self.classifier(slide_feature_vector)
            slide_outputs.append(prediction)
            
        if not slide_outputs:
            return torch.empty(0, self.classifier[-1].out_features).to(next(self.parameters()).device)

        return torch.cat(slide_outputs, dim=0)

def get_uni_transforms(model):
    """
    Funzione helper per ottenere le trasformazioni corrette per il modello UNI.
    """
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    return transform