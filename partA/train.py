import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from prepare_data import DataPreparation
from model import CNNModel
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN model with command line arguments')
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401_assignment_2',
                        help='Weights & Biases project name')
    parser.add_argument('-we', '--wandb_entity', type=str, 
                        default='da24s009-indiam-institute-of-technology-madras',
                        help='Weights & Biases entity name')
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    # Model architecture parameters
    parser.add_argument('--base_filters', type=int, default=32,
                        help='Base number of filters')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of convolutional layers')
    parser.add_argument('--filter_strategy', type=str, default='doubling',
                        choices=['same', 'doubling', 'halving'],
                        help='Filter count strategy')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'GELU', 'SiLU'],
                        help='Activation function')
    parser.add_argument('--dense_neurons', type=int, default=512,
                        help='Number of neurons in dense layer')
    parser.add_argument('--conv_dropout', type=float, default=0.0,
                        help='Dropout rate for convolutional layers')
    parser.add_argument('--dense_dropout', type=float, default=0.4,
                        help='Dropout rate for dense layers')
    
    return parser.parse_args()

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Move model to GPU
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Watch model in wandb
    wandb.watch(model, criterion, log="all")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
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
    if strategy == 'same':
        return [base_filters] * n_layers
    elif strategy == 'doubling':
        return [base_filters * (2**i) for i in range(n_layers)]
    elif strategy == 'halving':
        return [base_filters // (2**i) for i in range(n_layers)]
    else:
        return [base_filters] * n_layers

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args
    )
    
    # Load dataset with specified batch size
    data_directory = "../inaturalist_12K"
    data_preparation = DataPreparation(data_directory, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = data_preparation.get_data_loaders()
    
    # Get filter configuration
    conv_filters = get_filter_counts(
        args.base_filters,
        args.n_layers,
        args.filter_strategy
    )
    
    # Create model with command line parameters
    model = CNNModel(
        num_classes=10,
        conv_filters=conv_filters,
        filter_size=3,
        dense_neurons=[args.dense_neurons],
        conv_activation=getattr(nn, args.activation),
        dense_activation=getattr(nn, args.activation),
        conv_dropout_rate=args.conv_dropout,
        dense_dropout_rate=args.dense_dropout
    )
    
    # Train model
    train_model(model, train_loader, val_loader, args)