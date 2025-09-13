import numpy as np
from src.preprocessing import load_and_preprocess
from sklearn.datasets import load_wine

def test_no_missing_values():
    data = load_wine()
    assert not np.isnan(data.data).any(), "❌ Dataset mengandung missing values!"

def test_class_balance():
    data = load_wine()
    class_counts = np.bincount(data.target)
    imbalance_ratio = max(class_counts) / min(class_counts)
    assert imbalance_ratio < 3, f"❌ Dataset terlalu imbalance (ratio={imbalance_ratio:.2f})"

def test_data_not_empty():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
