from src.preprocessing import load_and_preprocess
from src.model import train_model, save_model
from src.evaluate import evaluate_model
import json

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    print(f"Akurasi model: {acc:.4f}")

    # save metrics
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    # ✅ konsisten simpan di root project
    save_model(model, "best_model.joblib")
    print("✅ Model terbaik disimpan ke 'best_model.joblib'")
