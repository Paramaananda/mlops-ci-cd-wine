# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/winequality-red.csv"):
    df = pd.read_csv(path, sep=";")

    # ubah target ke binary classification
    y = (df["quality"] >= 6).astype(int)
    X = df.drop("quality", axis=1)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
