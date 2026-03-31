# =========================
# IMPORTS
# =========================
import matplotlib.pyplot as plt
import os


# =========================
# CREATE RESULTS DIRECTORY
# =========================
def ensure_dir():
    os.makedirs("results", exist_ok=True)


# =========================
# PLOT SINGLE TRAINING
# =========================
def plot_single(train_losses, test_losses, title, filename):
    ensure_dir()

    epochs = range(1, len(train_losses) + 1)

    plt.figure()

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    plt.savefig(f"results/{filename}")
    plt.close()


# =========================
# PLOT COMPARISON
# =========================
def plot_comparison(normal_train, normal_test,
                    tuned_train, tuned_test):

    ensure_dir()

    epochs = range(1, len(normal_train) + 1)

    plt.figure()

    # Normal
    plt.plot(epochs, normal_train, label="Train (Normal)")
    plt.plot(epochs, normal_test, label="Test (Normal)")

    # Tuned
    plt.plot(epochs, tuned_train, '--', label="Train (Tuned)")
    plt.plot(epochs, tuned_test, '--', label="Test (Tuned)")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Normal vs Bayesian Optimization")
    plt.legend()

    plt.savefig("results/comparison.png")
    plt.close()