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
        model.train() # Make sure model is in training mode
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
        model.eval() # Switch to evaluation mode
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
            # Save model checkpoint only if it's the best so far
            # Consider saving to a unique path per run if needed
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            # wandb.save('best_model_resnet.pth') # Optional: save to wandb artifacts

def sweep_train():
    # Load dataset
    data_directory = "../inaturalist_12K"
    # Consider adding data augmentation here if not already done
    data_preparation = DataPreparation(data_directory, batch_size=32)
    train_loader, val_loader = data_preparation.get_data_loaders()

    def train():
        with wandb.init() as run:
            config = wandb.config

            # Determine freeze_upto_stage based on config
            # Handle the case where config.freeze_strategy might be 'all' or 'none'
            freeze_stage = None
            if config.freeze_strategy == 'upto_stage_1':
                freeze_stage = 1
            elif config.freeze_strategy == 'upto_stage_2':
                freeze_stage = 2
            elif config.freeze_strategy == 'upto_stage_3':
                freeze_stage = 3
            # 'none' corresponds to freeze_stage = None (fine-tune all)

            # Updated model creation for ResNet with freezing strategy
            model = ResNetModel(
                num_classes=10,
                dropout_rate=config.dropout_rate,
                freeze_upto_stage=freeze_stage # Pass the calculated stage
            )

            # Updated run name for ResNet parameters including freeze strategy
            run.name = f"freeze_{config.freeze_strategy}_dr{config.dropout_rate}_lr{round(config.learning_rate,5)}_ep{config.epochs}"

            train_model(model, train_loader, val_loader, config)

    return train

if __name__ == "__main__":
    # Updated sweep configuration for ResNet with freezing strategies
    sweep_config = {
        'method': 'bayes', # Or 'random'
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        # 'early_terminate': { # Optional: Add early stopping if desired
        #     'type': 'hyperband',
        #     'min_iter': 5, # Min epochs before stopping
        #     'eta': 2
        # },
        'parameters': {
            'dropout_rate': {'values': [0.3, 0.5]},
            'freeze_strategy': {'values': ['none', 'upto_stage_1', 'upto_stage_2', 'upto_stage_3']}, # 'none' means fine-tune all
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3}, # Log scale for LR
            'epochs': {'distribution': 'int_uniform', 'min': 2, 'max': 10} # Adjust epochs as needed
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        entity='da24s009-indiam-institute-of-technology-madras', # Replace with your entity if different
        project="da6401_assignment_2_resnet"
    )

    # Run sweep
    wandb.agent(sweep_id, function=sweep_train(), count=15)