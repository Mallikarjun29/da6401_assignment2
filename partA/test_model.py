import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from model import CNNModel
from prepare_data import DataPreparation

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
data_directory = "../inaturalist_12K"
data_preparation = DataPreparation(data_directory, batch_size=32)
_, _, test_loader = data_preparation.get_data_loaders()

# Load best model (update path if needed)
model = CNNModel(num_classes=10)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Evaluate on test set
correct = 0
total = 0
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())

test_acc = 100. * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# Prepare 10x3 grid of sample images and predictions
num_rows = 10
num_cols = 3
num_samples = num_rows * num_cols

images = torch.cat(all_images, dim=0)[:num_samples]
preds = torch.cat(all_preds, dim=0)[:num_samples]
labels = torch.cat(all_labels, dim=0)[:num_samples]

# Unnormalize images if needed (assuming mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*2.5))
fig.suptitle('Test Samples and Predictions (Best Model)', fontsize=18, fontweight='bold', color='navy')

for i in range(num_rows):
    for j in range(num_cols):
        idx = i * num_cols + j
        ax = axes[i, j]
        img = unnormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        pred = preds[idx].item()
        label = labels[idx].item()
        ax.set_title(f"True: {label}\nPred: {pred}", fontsize=10, fontweight='bold', color=('green' if pred==label else 'red'))
        ax.axis('off')

# Add a creative border and background color
def set_grid_style(fig, axes):
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#f0f8ff')
            for spine in ax.spines.values():
                spine.set_edgecolor('#4682b4')
                spine.set_linewidth(2)
set_grid_style(fig, axes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.savefig('test_predictions_grid.png', dpi=200)
plt.show()
