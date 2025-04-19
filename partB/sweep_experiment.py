"""
sweep_experiment.py

This script performs hyperparameter optimization for the ResNetModel using Weights & Biases (wandb).
It includes functions for training the model, logging metrics, and running sweeps with different configurations.

Functions:
    train_model: Trains the ResNetModel with specified configurations and logs metrics.
    sweep_train: Sets up the sweep configuration and runs the training process.

Example Usage:
    # Run the script to start a sweep
    python sweep_experiment.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from prepare_data import DataPreparation
from model import ResNetModel
import numpy as np

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, config):
    """
    Trains the ResNetModel and evaluates it on the validation set.

    Args:
        model (nn.Module): The ResNetModel to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        config (wandb.Config): Configuration object containing hyperparameters.

    Logs:
        Metrics such as training loss, validation loss, and accuracy to wandb.
    """
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer Setup: Only optimize trainable parameters ---
    # Filter parameters that require gradients
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config.learning_rate)
    # --- End Optimizer Setup ---

    # Move model to GPU
    model = model.to(device)
    criterion = criterion.to(device)

    # Watch model in wandb
    wandb.watch(model, criterion, log="all")

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        # Training phase
        model.train()  # Make sure model is in training mode
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
        model.eval()  # Switch to evaluation mode
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
            "train_loss": running_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model checkpoint only if it's the best so far
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            # wandb.save('best_model_resnet.pth') # Optional: save to wandb artifacts

def sweep_train():
    """
    Sets up the sweep configuration and runs the training process.

    Loads the dataset, initializes the ResNetModel with different configurations,
    and trains the model using wandb sweeps.

    Returns:
        function: A function to be used by wandb.agent for running sweeps.
    """
    # Load dataset
    data_directory = "../inaturalist_12K"
    data_preparation = DataPreparation(data_directory, batch_size=32)
    train_loader, val_loader, test_loader = data_preparation.get_data_loaders()

    def train():
        with wandb.init() as run:
            config = wandb.config

            # Determine freeze_upto_stage based on config
            freeze_stage = None
            if config.freeze_strategy == 'upto_stage_1':
                freeze_stage = 1
            elif config.freeze_strategy == 'upto_stage_2':
                freeze_stage = 2
            elif config.freeze_strategy == 'upto_stage_3':
                freeze_stage = 3

            # Initialize model with the specified freeze strategy
            model = ResNetModel(
                num_classes=10,
                dropout_rate=config.dropout_rate,
                freeze_upto_stage=freeze_stage
            )

            # Set run name for wandb
            run.name = f"freeze_{config.freeze_strategy}_dr{config.dropout_rate}_lr{round(config.learning_rate, 5)}_ep{config.epochs}"

            train_model(model, train_loader, val_loader, config)

    return train

if __name__ == "__main__":
    """
    Main entry point for the script.

    Sets up the sweep configuration and starts the wandb agent to perform hyperparameter optimization.
    """
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'dropout_rate': {'values': [0.3, 0.5]},
            'freeze_strategy': {'values': ['none', 'upto_stage_1', 'upto_stage_2', 'upto_stage_3']},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
            'epochs': {'distribution': 'int_uniform', 'min': 2, 'max': 10}
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        entity='da24s009-indiam-institute-of-technology-madras',
        project="da6401_assignment_2_resnet"
    )

    # Run sweep
    wandb.agent(sweep_id, function=sweep_train(), count=15)