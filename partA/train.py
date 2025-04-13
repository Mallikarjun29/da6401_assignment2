import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from prepare_data import DataPreparation
from model import CNNModel
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {running_loss/len(train_loader):.3f}, Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.3f}, Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load dataset
        data_directory = "/home/mallikarjun/da6401_assignment2/inaturalist_12K"
        data_preparation = DataPreparation(data_directory, batch_size=32)
        train_loader, val_loader = data_preparation.get_data_loaders()
        
        num_classes = len(train_loader.dataset.dataset.classes)
        print(f"\nNumber of classes: {num_classes}")
        print("Class names:", train_loader.dataset.dataset.classes)
        
        # Create model with memory-efficient settings
        model = CNNModel(
            num_classes=10,
            input_channels=3,
            conv_filters=[32, 64, 128, 256],  # Custom filter sizes
            filter_size=3,  # Larger filter size
            pool_size=2,
            dense_neurons=[512],  # Two dense layers
            conv_activation=nn.ReLU,  # Can be changed to nn.LeakyReLU etc.
            dense_activation=nn.ReLU,
            dropout_rate=0.5
        )
        
        # Train model
        train_model(model, train_loader, val_loader, num_epochs=10, device=device)
        
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        print("Try reducing batch size or model size further if memory issues persist")