from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(test_size=0.2, random_state=42):
    wine = load_wine()
    X, y = wine.data, wine.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler
