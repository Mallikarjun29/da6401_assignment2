import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(nn.Module):
    def __init__(self,
                 num_classes=10,
                 dropout_rate=0.5,
                 freeze_upto_stage=None): # Added parameter: None means fine-tune all
        super(ResNetModel, self).__init__()

        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # --- Freezing Logic ---
        if freeze_upto_stage is not None:
            # Freeze initial layers first
            for name, child in self.resnet.named_children():
                if name in ['conv1', 'bn1', 'relu', 'maxpool']:
                    for param in child.parameters():
                        param.requires_grad = False
                # Freeze specified stages (layer1, layer2, etc.)
                elif name.startswith('layer') and int(name.replace('layer', '')) <= freeze_upto_stage:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    # Unfreeze layers beyond the specified stage
                    for param in child.parameters():
                        param.requires_grad = True
        else:
             # Default: Fine-tune all layers (except potentially the classifier if needed)
             # Ensure all backbone layers are trainable if freeze_upto_stage is None
             for param in self.resnet.parameters():
                 param.requires_grad = True
        # --- End Freezing Logic ---

        # Get number of features from last layer
        num_features = self.resnet.fc.in_features

        # Replace final fully connected layer (always make this trainable)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Added dropout here too for consistency
            nn.Linear(512, num_classes)
        )
        # Ensure the new classifier head is trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True


    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Example Usage ---
    print("\n--- Example: Freeze up to Layer 2 ---")
    model_freeze_2 = ResNetModel(
        num_classes=10,
        dropout_rate=0.5,
        freeze_upto_stage=2 # Freeze conv1, bn1, relu, maxpool, layer1, layer2
    ).to(device)

    trainable_params_freeze_2 = sum(p.numel() for p in model_freeze_2.parameters() if p.requires_grad)
    print(f"Trainable parameters (freeze up to stage 2): {trainable_params_freeze_2:,}")
    # print(model_freeze_2) # Optional: print full model

    print("\n--- Example: Fine-tune all (freeze_upto_stage=None) ---")
    model_fine_tune_all = ResNetModel(
        num_classes=10,
        dropout_rate=0.5,
        freeze_upto_stage=None # Fine-tune everything
    ).to(device)

    trainable_params_fine_tune_all = sum(p.numel() for p in model_fine_tune_all.parameters() if p.requires_grad)
    print(f"Trainable parameters (fine-tune all): {trainable_params_fine_tune_all:,}")

    # Test with random input
    # x = torch.randn(1, 3, 224, 224).to(device)
    # output = model_freeze_2(x)
    # print(f"\nOutput shape: {output.shape}")