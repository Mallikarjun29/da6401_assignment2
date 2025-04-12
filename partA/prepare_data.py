import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class INaturalistDataset:
    """
    A class to handle the iNaturalist dataset preparation.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        
    def inspect_directory(self, dir_path):
        """
        Inspects directory contents and prints detailed information about files.
        """
        print(f"\nInspecting directory: {dir_path}")
        if not os.path.exists(dir_path):
            print("Directory does not exist!")
            return
            
        files = os.listdir(dir_path)
        if not files:
            print("Directory is empty!")
            return
            
        print(f"Found {len(files)} items in directory")
        extensions = {}
        for file in files:
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path):
                ext = os.path.splitext(file)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                
        print("\nFile extensions found:")
        for ext, count in extensions.items():
            print(f"{ext}: {count} files")
            
        if not extensions:
            print("No files found (might be subdirectories)")
            # List directory contents
            print("\nDirectory contents:")
            for item in files:
                print(f"- {item} ({'dir' if os.path.isdir(os.path.join(dir_path, item)) else 'file'})")

    def validate_dataset(self, subset):
        """
        Validates the dataset structure and prints information about missing or invalid files.
        """
        subset_dir = os.path.join(self.data_dir, subset)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        
        if not os.path.exists(subset_dir):
            print(f"Error: {subset} directory not found at {subset_dir}")
            return False
            
        for class_name in os.listdir(subset_dir):
            class_dir = os.path.join(subset_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            valid_files = False
            for file in os.listdir(class_dir):
                if file.lower().endswith(valid_extensions):
                    valid_files = True
                    break
            
            if not valid_files:
                print(f"\nWarning: No valid images found in class '{class_name}' directory")
                print(f"Path: {class_dir}")
                print(f"Supported extensions are: {', '.join(valid_extensions)}")
                self.inspect_directory(class_dir)

    def get_dataset(self, subset):
        """
        Returns the dataset for a specific subset (train/val).

        Args:
            subset (str): Subset of the dataset ('train' or 'val').

        Returns:
            torchvision.datasets.ImageFolder: The dataset for the specified subset.
        """
        self.validate_dataset(subset)
        subset_dir = os.path.join(self.data_dir, subset)
        return datasets.ImageFolder(subset_dir, transform=self.transform)


class DataPreparation:
    """
    A class to prepare data loaders for the iNaturalist dataset.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        """
        Initializes the data preparation class.

        Args:
            data_dir (str): Path to the dataset directory.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define data transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_data_loaders(self):
        """
        Prepares and returns data loaders for training and validation.

        Returns:
            train_loader, val_loader: Data loaders for training and validation.
        """
        dataset = INaturalistDataset(self.data_dir, transform=self.transform)
        train_dataset = dataset.get_dataset('train')
        val_dataset = dataset.get_dataset('val')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    data_directory = "/home/mallikarjun/da6401_assignment2/inaturalist_12K"
    dataset = INaturalistDataset(data_directory)
    
    print("\nValidating training dataset structure:")
    dataset.validate_dataset('train')
    print("\nValidating validation dataset structure:")
    dataset.validate_dataset('val')
    
    try:
        data_preparation = DataPreparation(data_directory)
        train_loader, val_loader = data_preparation.get_data_loaders()
        print(f"\nNumber of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print("\nClass labels (subfolder names):")
        print(train_loader.dataset.classes)
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")