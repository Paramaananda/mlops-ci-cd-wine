from src.model import train_model
from src.preprocessing import load_and_preprocess
from src.evaluate import evaluate_model

def test_metrics_are_in_valid_range():
    """Pastikan semua metrik ada dan dalam range [0,1]."""
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict), "Hasil evaluasi harus berupa dict"
    for key in ["accuracy", "f1_score", "precision", "recall"]:
        assert key in metrics, f"{key} tidak ditemukan di hasil evaluasi"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} berada di luar range [0,1]"

def test_model_minimum_performance():
    """Pastikan model mencapai threshold minimum akurasi & f1-score."""
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["accuracy"] >= 0.6, f"Akurasi terlalu rendah: {metrics['accuracy']:.4f}"
    assert metrics["f1_score"] >= 0.5, f"F1 Score terlalu rendah: {metrics['f1_score']:.4f}"

