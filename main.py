# =========================
# IMPORTS
# =========================
from torch.utils.data import DataLoader

from src.dataset import BikeDataset
from src.model import LSTMRegressor
from src.train import train_model
from src.optuna_tuning import run_optimization
from src.plot import plot_single, plot_comparison


# =========================
# HYPERPARAMETERS (BASELINE)
# =========================
INPUT_SIZE = 12
BATCH_SIZE = 32
HIDDEN_SIZE = 16
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 0.001
EPOCHS = 5


# =========================
# LOAD DATASET
# =========================
train_dataset = BikeDataset("data/X_train.npy", "data/y_train.npy")
test_dataset = BikeDataset("data/X_test.npy", "data/y_test.npy")


# =========================
# DATALOADERS (BASELINE)
# =========================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# 1. NORMAL TRAINING
# =========================
print("\n Starting Normal Training...\n")

model_normal = LSTMRegressor(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT
)

normal_train, normal_test = train_model(
    model_normal,
    train_loader,
    test_loader,
    EPOCHS,
    LEARNING_RATE
)

# Plot Normal
plot_single(
    normal_train,
    normal_test,
    title="Normal Training",
    filename="normal_training.png"
)


# =========================
# 2. OPTUNA OPTIMIZATION
# =========================
print("\n Starting Optuna Optimization...\n")

best_params, best_loss = run_optimization(
    train_dataset,
    test_dataset,
    n_trials=20
)

print("\nBest Parameters Found:")
for k, v in best_params.items():
    print(f"{k}: {v}")


# =========================
# 3. TRAIN WITH BEST PARAMS
# =========================
print("\n Training with Best Parameters...\n")

# Extract parameters
best_hidden = best_params["hidden_size"]
best_layers = best_params["num_layers"]
best_lr = best_params["learning_rate"]
best_batch = best_params["batch_size"]
best_dropout = best_params["dropout"]

if best_layers == 1:
    best_dropout = 0.0

# New DataLoader with best batch size
train_loader_tuned = DataLoader(train_dataset, batch_size=best_batch, shuffle=True)
test_loader_tuned = DataLoader(test_dataset, batch_size=best_batch, shuffle=False)

# New model
model_tuned = LSTMRegressor(
    INPUT_SIZE,
    best_hidden,
    best_layers,
    best_dropout
)

tuned_train, tuned_test = train_model(
    model_tuned,
    train_loader_tuned,
    test_loader_tuned,
    EPOCHS,
    best_lr
)

# Plot Tuned
plot_single(
    tuned_train,
    tuned_test,
    title="Bayesian Optimized Training",
    filename="tuned_training.png"
)


# =========================
# 4. COMPARISON PLOT
# =========================
plot_comparison(
    normal_train, normal_test,
    tuned_train, tuned_test
)


print("\n All Done! Check 'results/' folder for plots.")