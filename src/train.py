# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_loader, test_loader, epochs, lr):
    """
    Train model and return loss history
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- EVALUATE --------
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                test_loss += loss.item()

        test_loss /= len(test_loader)

        # store history
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")

    return train_losses, test_losses