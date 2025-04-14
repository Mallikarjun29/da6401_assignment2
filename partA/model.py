import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 input_channels=3,
                 conv_filters=[32, 32, 32, 32, 32],
                 filter_size=3,
                 pool_size=2,
                 dense_neurons=[512],
                 conv_activation=nn.ReLU,
                 dense_activation=nn.ReLU,
                 conv_dropout_rate=0.1,    # Lower dropout for conv layers
                 dense_dropout_rate=0.5):  # Higher dropout for dense layers
        super(CNNModel, self).__init__()
        
        self.features = nn.ModuleList()
        in_channels = input_channels
        
        # Create convolutional blocks with configurable parameters
        for out_channels in conv_filters:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=filter_size, 
                         padding='same'),
                conv_activation(),
                nn.Dropout2d(p=conv_dropout_rate),  # Spatial dropout after activation
                nn.MaxPool2d(pool_size)
            )
            self.features.append(conv_block)
            in_channels = out_channels
            
        # Adaptive pooling to get fixed size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Create dense layers
        self.classifier = nn.ModuleList()
        prev_neurons = conv_filters[-1] * 4 * 4
        
        for neurons in dense_neurons:
            dense_block = nn.Sequential(
                nn.Linear(prev_neurons, neurons),
                dense_activation(),
                nn.Dropout(p=dense_dropout_rate)  # Regular dropout for dense layers
            )
            self.classifier.append(dense_block)
            prev_neurons = neurons
            
        # Final classification layer (no dropout before output)
        self.classifier.append(nn.Linear(prev_neurons, num_classes))
    
    def forward(self, x):
        # Forward through convolutional blocks
        for feature_block in self.features:
            x = feature_block(x)
        
        # Final layers
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        # Forward through dense layers
        for classifier_block in self.classifier:
            x = classifier_block(x)
        
        return x

if __name__ == "__main__":
    # Example usage with custom parameters
    model = CNNModel(
        num_classes=10,
        input_channels=3,
        conv_filters=[32, 64, 128, 256, 512],  # Custom filter sizes
        filter_size=3,  # Larger filter size
        pool_size=2,
        dense_neurons=[1024, 512],  # Two dense layers
        conv_activation=nn.ReLU,  # Can be changed to nn.LeakyReLU etc.
        dense_activation=nn.ReLU,
        conv_dropout_rate=0.1,    # Example conv dropout rate
        dense_dropout_rate=0.5    # Example dense dropout rate
    )
    
    # Print model architecture
    print(model)
    
    # Test with random input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")