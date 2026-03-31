# =========================
# IMPORTS
# =========================
import optuna
import torch.nn as nn
import torch.optim as optim
from src.model import LSTMRegressor


# =========================
# OBJECTIVE FUNCTION
# =========================
def objective(trial, train_dataset, test_dataset):

    # 🔷 SEARCH SPACE
    hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    if num_layers == 1:
        dropout = 0.0


    # 🔷 MODEL
    model = LSTMRegressor(
        input_size=12,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # 🔷 LOSS & OPTIMIZER
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 🔷 TRAIN (SHORT)
    EPOCHS = 3
    for _ in range(EPOCHS):
        model.train()

        for X_batch, y_batch in train_dataset:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 🔷 EVALUATE
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_dataset:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataset)

    return avg_loss


# =========================
# RUN OPTUNA
# =========================
def run_optimization(train_dataset, test_dataset, n_trials=20):

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, train_dataset, test_dataset),
        n_trials=n_trials
    )

    print("\n✅ Optimization Finished")
    print(f"Best Loss: {study.best_value:.4f}")
    print("Best Params:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    return study.best_params, study.best_value