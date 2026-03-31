

# =========================
# IMPORTS
# =========================
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# =========================
# CUSTOM DATASET CLASS
# =========================
class BikeDataset(Dataset):
    """
    PyTorch Dataset for bike rental sequences.
    Each sample is a sequence of time steps and target cnt.
    """

    def __init__(self, X_path, y_path):
        """
        Args:
            X_path (str): Path to X numpy file (preprocessed sequences)
            y_path (str): Path to y numpy file
        """
        # Load preprocessed numpy arrays
        self.X = np.load(X_path)
        self.y = np.load(y_path)

        # Convert to PyTorch tensors (float for features, float or long for target)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)  # regression target

    def __len__(self):
        # Total number of sequences
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sequence and corresponding target
        return self.X[idx], self.y[idx]

# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Paths to preprocessed files
    X_train_path = "data/X_train.npy"
    y_train_path = "data/y_train.npy"
    X_test_path = "data/X_test.npy"
    y_test_path = "data/y_test.npy"

    # Create dataset objects
    train_dataset = BikeDataset(X_train_path, y_train_path)
    test_dataset = BikeDataset(X_test_path, y_test_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check shapes
    for X_batch, y_batch in train_loader:
        print(f"Batch X shape: {X_batch.shape}")  # (batch_size, time_steps, num_features)
        print(f"Batch y shape: {y_batch.shape}")  # (batch_size,)
        break  # just show first batch