# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# =========================
# LOAD DATA
# =========================
def load_data(path):
    return pd.read_csv(path)


# =========================
# SORT DATA BY TIME
# =========================
def sort_data(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    df = df.sort_values(by=['dteday', 'hr'])
    df = df.reset_index(drop=True)
    return df


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df.drop(columns=['instant', 'casual', 'registered', 'dteday'])
    return df


# =========================
# SPLIT FEATURES & TARGET
# =========================
def split_features_target(df):
    X = df.drop(columns=['cnt'])
    y = df['cnt']
    return X, y


# =========================
# TRAIN-TEST SPLIT (NO SHUFFLE)
# =========================
def train_test_split(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, y_train, X_test, y_test


# =========================
# SCALE FEATURES (TRAIN ONLY)
# =========================
def scale_features(X_train,y_train, X_test, y_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled,  y_train_scaled, X_test_scaled, y_test_scaled


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(X, y, time_steps=24):
    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


# =========================
# SAVE DATA
# =========================
def save_data(X_train, y_train, X_test, y_test):
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading data...")
    df = load_data("data/hour.csv")

    print("Sorting data...")
    df = sort_data(df)

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    print("Splitting train and test...")
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    print("Scaling features (train only)...")
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_features(X_train, y_train, X_test, y_test)

    print("Creating sequences...")
    time_steps = 24

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

    print("Saving processed data...")
    save_data(X_train_seq, y_train_seq, X_test_seq, y_test_seq)

    print("\n✅ Preprocessing completed!")
    print(f"X_train shape: {X_train_seq.shape}")
    print(f"y_train shape: {y_train_seq.shape}")
    print(f"X_test shape: {X_test_seq.shape}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()