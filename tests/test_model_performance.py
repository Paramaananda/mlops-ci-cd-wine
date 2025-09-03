from src.model import train_model
from src.preprocessing import load_and_preprocess
from src.evaluate import evaluate_model

def test_model_minimum_accuracy():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    assert acc >= 0.7, f"Akurasi terlalu rendah: {acc}"
