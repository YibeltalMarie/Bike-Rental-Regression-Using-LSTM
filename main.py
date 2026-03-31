
import torch
from torch.utils.data import DataLoader

from src.dataset import BikeDataset
from src.model import LSTMRegressor
from src.train import train_model


# hyperparameters
BATCH_SIZE = 32
INPUT_SIZE = 12
HIDDEN_SIZE = 16
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 0.001
EPOCHS = 5

# dataset
train_dataset = BikeDataset("data/X_train.npy", "data/y_train.npy")
test_dataset = BikeDataset("data/X_test.npy", "data/y_test.npy")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model
model = LSTMRegressor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
# train
train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)