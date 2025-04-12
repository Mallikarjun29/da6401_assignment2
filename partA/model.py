import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 input_channels=3,
                 num_conv_blocks=5,
                 filters_per_layer=64,
                 filter_size=(3,3),
                 pool_size=(2,2),
                 dense_neurons=1024,
                 conv_activation=nn.ReLU,
                 dense_activation=nn.ReLU):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential()
        current_channels = input_channels
        
        # Build conv-activation-maxpool blocks
        for i in range(num_conv_blocks):
            # Convolution layer
            self.features.add_module(f'conv{i+1}', 
                nn.Conv2d(current_channels, filters_per_layer, filter_size, padding='same'))
            
            # Activation layer
            self.features.add_module(f'act{i+1}', conv_activation())
            
            # Max pooling layer with small pool size to prevent excessive downsampling
            self.features.add_module(f'pool{i+1}', nn.MaxPool2d((1, 1)))
            
            current_channels = filters_per_layer
        
        # Add adaptive pooling to get fixed size output
        self.features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((4, 4)))
        
        # Flatten layer
        self.features.add_module('flatten', nn.Flatten())
        
        # Calculate the size of flattened features
        flattened_size = filters_per_layer * 4 * 4  # Fixed size from adaptive pooling
        
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, dense_neurons),
            dense_activation(),
            nn.Dropout(0.5),
            nn.Linear(dense_neurons, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Example usage
if __name__ == "__main__":
    # Example configuration with 5 layers
    model = CNNModel(
        num_classes=10,
        input_channels=3,
        num_conv_blocks=5,
        filters_per_layer=64,
        filter_size=(3,3),
        pool_size=(2,2),
        dense_neurons=1024
    )
    
    # Print model architecture
    print(model)
    
    # Test with random input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")