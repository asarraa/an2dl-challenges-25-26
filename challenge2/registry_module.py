import os
import json
import datetime
import torch

class ModelRegistry:
    def __init__(self, base_dir="experiments"):
        """
        Args:
            base_dir: Folder where models and the registry.json will be saved.
        """
        self.base_dir = base_dir
        self.registry_path = os.path.join(base_dir, "registry.json")
        self.models_dir = os.path.join(base_dir, "models")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing registry or create new
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: {self.registry_path} was corrupted or empty. Starting a fresh registry.")
                self.registry = {}
        else:
            self.registry = {}

    def generate_id(self, prefix="exp"):
        """Generates a unique ID based on timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def save_experiment(self, model, optimizer, train_cfg, model_cfg, metrics,  run_id=None):
        """
        Saves the model weights and logs metadata to registry.json.
        
        Returns:
            experiment_id (str): The ID assigned to this run.
        """

         # 1. Determine ID
        if run_id is not None:
            exp_id = run_id
        else:
            exp_id = self.generate_id(prefix=model_cfg.get("model_name", "model"))
        
        # 2. Define File Paths
        model_filename = f"{exp_id}.pt"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # 3. Save PyTorch Model
        # It's good practice to save optimizer state too for resuming
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {**train_cfg, **model_cfg}, # Save config inside .pt too for safety
            'metrics': metrics
        }
        torch.save(save_dict, model_path)
        
        # 4. Update Registry JSON
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_path": model_path,
            "training_params": train_cfg,
            "model_architecture": model_cfg,
            "final_metrics": metrics
        }
        
        self.registry[exp_id] = entry
        
        # Write back to disk
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
            
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Registry updated: ID {exp_id}")
        
        return exp_id
    

def load_model(self, exp_id, model_class, device):
        """Helper to load a model by ID"""
        if exp_id not in self.registry:
            raise ValueError(f"ID {exp_id} not found in registry.")
            
        entry = self.registry[exp_id]
        path = entry["model_path"]
        config = entry["model_architecture"]
        
        # Instantiate
        model = model_class(**config)
        
        # Load Weights
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(device)