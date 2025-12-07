# Set seed for reproducibility
SEED = 42

import numpy as np
import torch
torch.manual_seed(SEED)
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)
#%matplotlib inline


# @title Activation visualisation
def get_activation(name):
    """Creates a hook function to capture and store layer outputs."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def find_last_conv_layer(model):
    """
    Identifies the final Conv2D layer in the model architecture.
    """
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name

    if last_conv_name is None:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv_name


def visualize(model, X, y, unique_labels, num_images=50, display_activations=True, display_all_conv_layers=False, device_obj = None):
    """
    Visualises model predictions and internal activations for a random test image.
    Uses PyTorch hooks to extract intermediate layer outputs.

    Args:
        display_all_conv_layers: If True, shows all conv layers. If False, shows only last conv of each block.
    """

    # --- 1. Select Image and Prepare Tensor ---

    # Randomly select an image from the dataset
    image_idx = np.random.randint(0, num_images)
    img_np = X[image_idx]
    label_np = y[image_idx]

    # Convert NumPy array to PyTorch tensor with correct dimensions
    # Transform from (H, W, C) to (N, C, H, W) format
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).to(device_obj)

    # --- 2. Register Hooks and Make Prediction ---

    # Clear previous activations
    activations.clear()

    # Attach forward hooks to convolutional layers
    hooks = []
    conv_names = []

    # Iterate through all blocks in the features Sequential
    for block_idx, block in enumerate(model.features):
        # Find all Conv2d layers in this block
        conv_layers_in_block = []
        for layer_idx, layer in enumerate(block.block):
            if isinstance(layer, nn.Conv2d):
                conv_layers_in_block.append((layer_idx, layer))

        # Register hooks based on display_all_conv_layers flag
        if display_all_conv_layers:
            # Register hook for every Conv2d layer
            for layer_idx, conv_layer in conv_layers_in_block:
                hook_name = f'block{block_idx}_conv{layer_idx}'
                conv_names.append(hook_name)
                hooks.append(conv_layer.register_forward_hook(get_activation(hook_name)))
        else:
            # Register hook only for the last Conv2d layer in this block
            if conv_layers_in_block:
                layer_idx, conv_layer = conv_layers_in_block[-1]
                hook_name = f'block{block_idx}_conv{layer_idx}'
                conv_names.append(hook_name)
                hooks.append(conv_layer.register_forward_hook(get_activation(hook_name)))

    # Generate prediction with gradient tracking disabled
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Remove hooks after forward pass
    for hook in hooks:
        hook.remove()

    # Extract predicted class and confidence
    predictions = probabilities.cpu().numpy()
    class_int = np.argmax(predictions[0])
    class_str = unique_labels[class_int]

    # Extract true class (handle both one-hot encoded and integer labels)
    if label_np.ndim > 0 and len(label_np) > 1:
        # One-hot encoded
        true_class_int = np.argmax(label_np)
    else:
        # Already an integer index
        true_class_int = int(label_np)
    true_class_str = unique_labels[true_class_int]

    # --- 3. Plot Image and Prediction Bar ---

    # Create figure with custom layout
    fig = plt.figure(constrained_layout=True, figsize=(16, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.5], wspace=0)

    # Display original image with true label
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f"True class: {true_class_str}", loc='left')
    if img_np.shape[-1] == 1:
        ax1.imshow(np.squeeze(img_np), cmap='bone', vmin=0., vmax=1.)
    else:
        ax1.imshow(np.squeeze(img_np), vmin=0., vmax=1.)
    ax1.axis('off')

    # Display class probability distribution
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(unique_labels, np.squeeze(predictions, axis=0), color=plt.get_cmap('tab10').colors)
    ax2.set_title(f"Predicted class: {class_str} (Confidence: {max(np.squeeze(predictions[0])):.2f})", loc='left')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0.0, 1.0)
    plt.show()

    # --- 4. Plot Activations ---

    if display_activations:
        # Visualise activations for each registered layer
        for conv_name in conv_names:
            # Retrieve stored activations from hooks
            layer_activations = activations[conv_name]

            # Get number of channels
            num_channels = layer_activations.shape[1]

            # Display up to 16 feature maps per layer
            num_display = min(16, num_channels)

            # Calculate grid layout
            if num_display <= 8:
                rows, cols = 1, num_display
                figsize = (18, 3)
            else:
                rows, cols = 2, 8
                figsize = (18, 5)

            # Create subplot grid
            fig, axes = plt.subplots(rows, cols, figsize=figsize)

            # Flatten axes array for easier indexing
            if num_display > 1:
                axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
            else:
                axes = [axes]

            # Plot each activation map
            for i in range(num_display):
                ax = axes[i]
                activation_map = layer_activations[0, i].cpu().numpy()
                ax.imshow(activation_map, cmap='bone', vmin=np.min(activation_map), vmax=np.max(activation_map))
                ax.axis('off')
                if i == 0:
                    ax.set_title(f'{conv_name} activations', loc='left')

            # Hide unused subplots
            for i in range(num_display, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()


def make_inference(best_model, test_loader, X_test, y_test, unique_labels, device_obj):

    # Dictionary to store layer activations via forward hooks
    activations = {}

    # Visualise model predictions and internal representations
    # Set display_all_conv_layers=True to show all conv layers, False for only last conv of each block
    visualize(best_model, X_test, y_test, unique_labels, display_activations=True, display_all_conv_layers=False, device_obj)


    # Collect predictions and ground truth labels
    test_preds, test_targets = [], []
    with torch.no_grad():  # Disable gradient computation for inference
        for xb in test_loader:
            xb = xb[0].to(device_obj)

            # Forward pass: get model predictions
            logits = best_model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()

            # Store batch results
            test_preds.append(preds)

    # Combine all batches into single arrays
    test_preds = np.concatenate(test_preds)


    # ✅ Dizionario che mappa le classi numeriche a quelle testuali
    label_map = {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3:"Triple Negative" }

    # ✅ Converte le predizioni numeriche nel formato testuale
    submission_data = []
    for uid, pred_num in test_preds.items():
        label_str = label_map[pred_num]
        submission_data.append((f"{int(uid):04d}", label_str))  # formatta l’ID come '000', '001', ecc.

    # ✅ Crea il DataFrame per la submission
    submission = pd.DataFrame(submission_data, columns=["sample_index", "label"])

    # ✅ Salva il CSV
    submission.to_csv("submission.csv", index=False)
    print("📁 File 'submission.csv' creato con successo!")

    # ✅ (Facoltativo) scarica il file in locale (solo in Colab)
    from google.colab import files
    files.download("submission.csv")


    # Calculate overall test accuracy
    '''
    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, average='weighted')
    test_rec = recall_score(test_targets, test_preds, average='weighted')
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    print(f"Accuracy over the test set: {test_acc:.4f}")
    print(f"Precision over the test set: {test_prec:.4f}")
    print(f"Recall over the test set: {test_rec:.4f}")
    print(f"F1 score over the test set: {test_f1:.4f}")'''

    # Generate confusion matrix for detailed error analysis
    #cm = confusion_matrix(test_targets, test_preds)

    # Create numeric labels for heatmap annotation
    '''
    labels = np.array([f"{num}" for num in cm.flatten()]).reshape(cm.shape)

    # Visualise confusion matrix
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=labels, fmt='',
                cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.show()
    '''