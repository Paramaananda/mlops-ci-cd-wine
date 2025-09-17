from src.model import train_model
from src.preprocessing import load_and_preprocess
from src.evaluate import evaluate_model

def test_evaluate_model_returns_dict():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict), "evaluate_model harus mengembalikan dict"

    for key in ["accuracy", "f1_score", "precision", "recall"]:
        assert key in metrics, f"{key} tidak ditemukan di hasil evaluasi"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} berada di luar range [0,1]"

def test_evaluate_model_minimum_accuracy():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["accuracy"] >= 0.5, f"Akurasi terlalu rendah: {metrics['accuracy']:.4f}"
