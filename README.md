# DA6401 Assignment 2

This repository contains the implementation of image classification tasks using CNN and ResNet models. The project is divided into two parts:

- **Part A**: Implementation of a configurable CNN model.
- **Part B**: Implementation of a ResNet50 model with transfer learning.

## Repo(rt) links
[WandB report](https://wandb.ai/da24s009-indiam-institute-of-technology-madras/da6401_assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjMxNjIyMw?accessToken=utstsd3p1fzg5tz2885w5eqee1k6az9e5o30bd5r0f6q7kpkrgg3rfmcleal95l7)

[github](https://github.com/Mallikarjun29/da6401_assignment2/tree/main)
## File Structure

```
/home/mallikarjun/da6401_assignment2
│
├── partA
│   ├── model.py                # Configurable CNN model implementation
│   ├── prepare_data.py         # Data preparation and validation utilities
│   ├── train.py                # Training script for CNN model
│   ├── test_model.py           # Testing and evaluation script for CNN model
│   ├── sweep_experiment.py     # Hyperparameter sweep script for CNN model
│   ├── wandb/                  # Weights & Biases logs and metadata
│
├── partB
│   ├── model.py                # ResNet50 model with transfer learning
│   ├── prepare_data.py         # Data preparation and validation utilities
│   ├── train.py                # Training script for ResNet50 model
│   ├── sweep_experiment.py     # Hyperparameter sweep script for ResNet50
│
├── README.md                   # Project documentation
├── .gitignore                  # Ignored files and directories
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mallikarjun29/da6401_assignment2.git
   cd da6401_assignment2
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Part A: CNN Model

1. **Train the CNN model**:
   ```bash
   python partA/train.py -e 15 -b 32 -lr 0.001 --base_filters 64 --n_layers 5 --filter_strategy doubling --activation ReLU --dense_neurons 512 --conv_dropout 0.0 --dense_dropout 0.4
   ```

2. **Test the CNN model**:
   ```bash
   python partA/test_model.py
   ```

3. **Run hyperparameter sweep for CNN model**:
   ```bash
   python partA/sweep_experiment.py
   ```

### Part B: ResNet50 Model

1. **Train the ResNet50 model**:
   ```bash
   python partB/train.py -e 10 -b 32 -lr 0.001 --freeze_strategy none --dropout_rate 0.5
   ```

2. **Run hyperparameter sweep for ResNet50 model**:
   ```bash
   python partB/sweep_experiment.py
   ```

## Code Descriptions

### Part A

- **`model.py`**: Implements a configurable CNN model with customizable layers, activation functions, and dropout rates.
- **`prepare_data.py`**: Handles data loading, preprocessing, and train/validation/test splitting for the iNaturalist dataset.
- **`train.py`**: Trains the CNN model with specified hyperparameters and logs metrics to Weights & Biases.
  - **Arguments**:
    - `-wp` / `--wandb_project`: Weights & Biases project name (default: `da6401_assignment_2`).
    - `-we` / `--wandb_entity`: Weights & Biases entity name (default: `da24s009-indiam-institute-of-technology-madras`).
    - `-e` / `--epochs`: Number of training epochs (default: `15`).
    - `-b` / `--batch_size`: Training batch size (default: `32`).
    - `-lr` / `--learning_rate`: Learning rate (default: `0.001`).
    - `--base_filters`: Base number of filters for the first convolutional layer (default: `32`).
    - `--n_layers`: Number of convolutional layers (default: `5`).
    - `--filter_strategy`: Strategy for filter progression (`same`, `doubling`, `halving`) (default: `doubling`).
    - `--activation`: Activation function (`ReLU`, `GELU`, `SiLU`) (default: `ReLU`).
    - `--dense_neurons`: Number of neurons in the dense layer (default: `512`).
    - `--conv_dropout`: Dropout rate for convolutional layers (default: `0.0`).
    - `--dense_dropout`: Dropout rate for dense layers (default: `0.4`).
- **`test_model.py`**: Evaluates the trained CNN model on the test dataset and generates metrics like accuracy, confusion matrix, and classification report.
- **`sweep_experiment.py`**: Performs hyperparameter optimization using Weights & Biases sweeps for the CNN model.

### Part B

- **`model.py`**: Implements a ResNet50 model with transfer learning and options to freeze specific layers.
- **`prepare_data.py`**: Similar to Part A, handles data loading and preprocessing for the iNaturalist dataset.
- **`train.py`**: Trains the ResNet50 model with transfer learning strategies and logs metrics to Weights & Biases.
  - **Arguments**:
    - `-wp` / `--wandb_project`: Weights & Biases project name (default: `da6401_assignment_2_resnet`).
    - `-we` / `--wandb_entity`: Weights & Biases entity name (default: `da24s009-indiam-institute-of-technology-madras`).
    - `-e` / `--epochs`: Number of training epochs (default: `10`).
    - `-b` / `--batch_size`: Training batch size (default: `32`).
    - `-lr` / `--learning_rate`: Learning rate (default: `0.001`).
    - `--freeze_strategy`: Layer freezing strategy (`none`, `upto_stage_1`, `upto_stage_2`, `upto_stage_3`) (default: `none`).
    - `--dropout_rate`: Dropout rate for the fully connected layers (default: `0.5`).
- **`sweep_experiment.py`**: Performs hyperparameter optimization using Weights & Biases sweeps for the ResNet50 model.

## Notes

- Ensure the iNaturalist dataset is placed in the repo as `inaturalist_12K/` directory which has `train` and `val` as subfolders.
- Weights & Biases (wandb) is used for experiment tracking. Set up your wandb account and login before running the scripts:
  ```bash
  wandb login
  ```

## Contact
For any issues or questions, please contact [Mallikarjun](mailto:da24s009@smail.iitm.ac.in).