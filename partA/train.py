import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from prepare_data import DataPreparation
from model import CNNModel
import numpy as np

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Move model to GPU
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Watch model in wandb
    wandb.watch(model, criterion, log="all")
    
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Move data to GPU
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
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": running_loss/len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')

def get_filter_counts(base_filters, n_layers, strategy):
    """Helper function to generate filter counts based on strategy"""
    if strategy == 'same':
        return [base_filters] * n_layers
    elif strategy == 'doubling':
        return [base_filters * (2**i) for i in range(n_layers)]
    elif strategy == 'halving':
        return [base_filters // (2**i) for i in range(n_layers)]
    else:
        return [base_filters] * n_layers

def sweep_train():
    # Load dataset
    data_directory = "/home/mallikarjun/da6401_assignment2/inaturalist_12K"
    data_preparation = DataPreparation(data_directory, batch_size=32)
    train_loader, val_loader = data_preparation.get_data_loaders()
    
    def train():
        with wandb.init() as run:  # Remove entity and project from here
            config = wandb.config
            
            conv_filters = get_filter_counts(
                config.base_filters, 
                config.n_layers, 
                config.filter_strategy
            )
            
            # Updated model creation with separate dropout rates
            model = CNNModel(
                num_classes=10,
                conv_filters=conv_filters,
                filter_size=3,
                dense_neurons=[config.dense_neurons],
                conv_activation=getattr(nn, config.activation),
                dense_activation=getattr(nn, config.activation),
                conv_dropout_rate=config.conv_dropout,
                dense_dropout_rate=config.dense_dropout
            )
            
            # Set run name without calling save
            run.name = f"bf{config.base_filters}_nl{config.n_layers}_fs{config.filter_strategy}_act{config.activation}_dn{config.dense_neurons}_cdr{config.conv_dropout}_ddr{config.dense_dropout}_lr{config.learning_rate}_ep{config.epochs}"
            
            train_model(model, train_loader, val_loader, config)

    return train
# bf3_nl4_fsdoubling_actGELU_dn256_cdr0_ddr0.5_lr0.001_ep20
if __name__ == "__main__":
    # Updated sweep configuration with early stopping
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'base_filters': {'values': [32, 64, 128]},
            'n_layers': {'values': [4, 5]},
            'filter_strategy': {'values': ['same', 'doubling', 'halving']},
            'activation': {'values': ['ReLU', 'GELU', 'SiLU']},
            'dense_neurons': {'values': [256, 512, 1024]},
            'conv_dropout': {'values': [0, 0.1]},
            'dense_dropout': {'values': [0.3, 0.4, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'epochs': {'values': [5, 10, 15, 20]}
        }
    }
    
    # Initialize sweep with entity and project
    sweep_id = wandb.sweep(
        sweep_config,
        entity='da24s009-indiam-institute-of-technology-madras',
        project="da6401_assignment_2"
    )
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train(), count=100)