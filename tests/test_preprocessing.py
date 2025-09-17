from src.preprocessing import load_and_preprocess

def test_preprocessing():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert scaler is not None