"""
test_model.py

This script evaluates the best CNNModel on the test dataset using metrics such as accuracy, confusion matrix, and classification report.
It also visualizes predictions and logs results to Weights & Biases (wandb).

Functions:
    unnormalize: Reverts the normalization applied to images for visualization.

Example Usage:
    # Run the script to evaluate the best model
    python test_model.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb
from model import CNNModel
from prepare_data import DataPreparation
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from wandb import plot

# Initialize wandb
api = wandb.Api()
entity = "da24s009-indiam-institute-of-technology-madras"
project = "da6401_assignment_2"

# Find the best run based on validation accuracy
print("Finding best model from wandb runs...")
runs = api.runs(f"{entity}/{project}")
best_run = None
best_val_acc = 0

for run in runs:
    if run.state == "finished":
        val_acc = run.summary.get("val_acc", 0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run = run

if best_run is None:
    raise ValueError("No completed runs found in wandb")

print(f"\nBest run found: {best_run.name}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Initialize model with exact architecture from checkpoint
model = CNNModel(
    num_classes=10,
    conv_filters=[3, 6, 12, 24],  # Based on weight shapes from error message
    filter_size=3,
    dense_neurons=[512],  # Based on classifier dimensions
    conv_activation=nn.ReLU,
    dense_activation=nn.ReLU,
    conv_dropout_rate=0.0,
    dense_dropout_rate=0.4
)

print("\nModel architecture:")
print(model)

# Download and load the best model weights
best_model_file = wandb.restore('best_model.pth', run_path=f"{entity}/{project}/{best_run.id}")
if best_model_file is None:
    raise ValueError("Could not download model file from wandb")

model.load_state_dict(torch.load(best_model_file.name, map_location=device))
model = model.to(device)
model.eval()

# Load test data
data_directory = "../inaturalist_12K"
data_preparation = DataPreparation(data_directory, batch_size=32)
_, _, test_loader = data_preparation.get_data_loaders()

# Evaluate on test set
all_preds = []
all_labels = []
all_images = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.append(images.cpu())

test_acc = 100. * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# Generate and plot confusion matrix with normalized values
plt.figure(figsize=(12, 10))
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=data_preparation.classes,
            yticklabels=data_preparation.classes)
plt.title('Normalized Confusion Matrix on Test Set\n'
          f'Overall Accuracy: {test_acc:.2f}%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)

# Initialize new wandb run for test results
wandb.init(
    project="da6401_assignment_2",
    entity="da24s009-indiam-institute-of-technology-madras",
    name=f"test_results_{best_run.name}",
    config={"best_model": best_run.name}
)

# Log confusion matrix image
wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

plt.show()

# Generate and display classification report with zero_division=1
report = classification_report(
    all_labels, 
    all_preds,
    target_names=data_preparation.classes,
    output_dict=True,
    zero_division=1  # Handle zero-division case
)

# Convert to DataFrame and format for better visualization
df_report = pd.DataFrame(report).transpose()
df_report = df_report.round(3) * 100  # Convert to percentages

# Add column headers
df_report.columns = ['Precision %', 'Recall %', 'F1-Score %', 'Support']
df_report['Support'] = df_report['Support'].astype(int)

print("\nClassification Report:")
print(df_report)

# Save enhanced classification report to CSV
df_report.to_csv('classification_report.csv')

# Log classification report as a table
wandb_table = wandb.Table(
    dataframe=df_report.reset_index().rename(columns={'index': 'Class'})
)
wandb.log({"classification_report": wandb_table})

# Calculate per-class accuracy
class_correct = np.diag(cm)
class_total = np.sum(cm, axis=1)
class_acc = class_correct / class_total * 100

# Print per-class accuracies
print("\nPer-class Accuracies:")
for i, (name, acc) in enumerate(zip(data_preparation.classes, class_acc)):
    print(f"{name}: {acc:.2f}%")

# Create visualization grid
num_rows, num_cols = 10, 3
num_samples = num_rows * num_cols

images = torch.cat(all_images, dim=0)[:num_samples]
# Fix tensor concatenation for predictions and labels
preds = torch.tensor(all_preds)[:num_samples]  # Remove unnecessary cat()
labels = torch.tensor(all_labels)[:num_samples]  # Remove unnecessary cat()

# Get class names
class_names = data_preparation.classes

def unnormalize(img):
    """
    Reverts the normalization applied to images for visualization.

    Args:
        img (torch.Tensor): Normalized image tensor.

    Returns:
        torch.Tensor: Unnormalized image tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

# Create figure with custom style
plt.style.use('seaborn-v0_8-darkgrid')  # Updated style name
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 25))
fig.suptitle('Test Set Predictions of Best Model', 
             fontsize=20, fontweight='bold', y=0.999)

for i in range(num_rows):
    for j in range(num_cols):
        idx = i * num_cols + j
        ax = axes[i, j]
        
        img = unnormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        
        pred = class_names[preds[idx]]
        true = class_names[labels[idx]]
        
        color = 'green' if pred == true else 'red'
        title = f'True: {true}\nPred: {pred}'
        ax.set_title(title, color=color, fontsize=10, pad=10)
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('test_predictions_grid.png', bbox_inches='tight', dpi=300, pad_inches=0.5)
plt.show()

# Save and log the test predictions grid
wandb.log({"test_predictions_grid": wandb.Image('test_predictions_grid.png')})

# Log test accuracy
wandb.log({"test_accuracy": test_acc})

# Close wandb run
wandb.finish()
