"""Configurable CNN model implementation for image classification tasks.

Provides a flexible CNN architecture with customizable:
- Convolutional layers and filters
- Activation functions
- Dropout rates
- Dense layer architecture
"""

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """Configurable Convolutional Neural Network.
    
    Features:
    - Variable number of conv layers with configurable filters
    - Adaptive pooling for flexible input sizes
    - Configurable dense layer architecture
    - Independent dropout rates for conv and dense layers
    """
    
    def __init__(self, 
                 num_classes=10,
                 input_channels=3,
                 conv_filters=[32, 32, 32, 32, 32],
                 filter_size=3,
                 pool_size=2,
                 dense_neurons=[512],
                 conv_activation=nn.ReLU,
                 dense_activation=nn.ReLU,
                 conv_dropout_rate=0.1,
                 dense_dropout_rate=0.5):
        """Initialize CNN architecture.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels
            conv_filters (list): Number of filters for each conv layer
            filter_size (int): Kernel size for conv layers
            pool_size (int): Size of max pooling windows
            dense_neurons (list): Number of neurons in each dense layer
            conv_activation: Activation function for conv layers
            dense_activation: Activation function for dense layers
            conv_dropout_rate (float): Dropout rate for conv layers
            dense_dropout_rate (float): Dropout rate for dense layers
        """
        super(CNNModel, self).__init__()
        
        self.features = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_filters:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=filter_size, 
                         padding='same'),
                conv_activation(),
                nn.Dropout2d(p=conv_dropout_rate),
                nn.MaxPool2d(pool_size)
            )
            self.features.append(conv_block)
            in_channels = out_channels
            
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.ModuleList()
        prev_neurons = conv_filters[-1] * 4 * 4
        
        for neurons in dense_neurons:
            dense_block = nn.Sequential(
                nn.Linear(prev_neurons, neurons),
                dense_activation(),
                nn.Dropout(p=dense_dropout_rate)
            )
            self.classifier.append(dense_block)
            prev_neurons = neurons
            
        self.classifier.append(nn.Linear(prev_neurons, num_classes))
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes)
        """
        for feature_block in self.features:
            x = feature_block(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        for classifier_block in self.classifier:
            x = classifier_block(x)
        
        return x

if __name__ == "__main__":
    """Test model initialization and forward pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNModel(
        num_classes=10,
        input_channels=3,
        conv_filters=[32, 64, 128, 256, 512],
        filter_size=3,
        pool_size=2,
        dense_neurons=[1024, 512],
        conv_activation=nn.ReLU,
        dense_activation=nn.ReLU,
        conv_dropout_rate=0.1,
        dense_dropout_rate=0.5
    ).to(device)
    
    print(model)
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")