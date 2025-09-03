import numpy as np
from sklearn.datasets import load_wine

def test_no_missing_values():
    """Pastikan dataset tidak ada missing value."""
    data = load_wine()
    assert not np.isnan(data.data).any(), "❌ Dataset mengandung missing values!"

def test_class_balance():
    """Pastikan dataset tidak terlalu imbalance."""
    data = load_wine()
    class_counts = np.bincount(data.target)
    imbalance_ratio = max(class_counts) / min(class_counts)
    assert imbalance_ratio < 3, f"❌ Dataset terlalu imbalance (ratio={imbalance_ratio:.2f})"

def test_feature_range():
    """Pastikan semua fitur bernilai positif (misal untuk wine dataset)."""
    data = load_wine()
    assert (data.data >= 0).all(), "❌ Ada nilai fitur negatif tidak wajar!"
