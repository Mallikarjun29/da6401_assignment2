import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 dropout_rate=0.5,
                 fine_tune=True):
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # print([f"{name}: {layer.__class__.__name__}" for name, layer in self.resnet.named_modules()])
        if not fine_tune:
            # Freeze all layers if not fine-tuning
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get number of features from last layer
        num_features = self.resnet.fc.in_features
        
        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = ResNetModel(
        num_classes=10,
        dropout_rate=0.5,
        fine_tune=True
    ).to(device)
    
    # Print model architecture
    # print(model)
    
    # Test with random input (ResNet expects 224x224 images)
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Print number of trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {trainable_params:,}")