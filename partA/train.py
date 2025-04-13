import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from prepare_data import DataPreparation
from model import CNNModel
import numpy as np

def train_model(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
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

def sweep_train():
    # Load dataset
    data_directory = "/home/mallikarjun/da6401_assignment2/inaturalist_12K"
    data_preparation = DataPreparation(data_directory, batch_size=32)
    train_loader, val_loader = data_preparation.get_data_loaders()
    
    def train():
        # Initialize wandb with sweep config
        with wandb.init(entity='da24s009-indiam-institute-of-technology-madras', 
                       project="da6401_assignment_2") as run:
            # Access sweep config
            config = wandb.config
            
            # Create model with sweep config
            model = CNNModel(
                num_classes=10,
                conv_filters=[config.n_filters] * config.n_layers,
                filter_size=3,
                dense_neurons=[config.dense_neurons],
                conv_activation=getattr(nn, config.activation),
                dropout_rate=config.dropout
            )
            
            # Train model
            train_model(model, train_loader, val_loader, config)

    return train

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'n_filters': {'values': [3, 3, 3]},
            'n_layers': {'values': [3, 4, 5]},
            'activation': {'values': ['ReLU', 'GELU', 'SiLU']},
            'dense_neurons': {'values': [256, 512, 1024]},
            'dropout': {'values': [0.2, 0.3, 0.5]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'epochs': {'value': 10}
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="da6401_assignment_2")
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train(), count=10)