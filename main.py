from src.preprocessing import load_and_preprocess
from src.model import train_model, save_model
from src.evaluate import evaluate_model
import json

if __name__ == "__main__":
    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # Step 2: Train model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate
    acc = evaluate_model(model, X_test, y_test)
    print(f"Akurasi model: {acc:.4f}")

    # Save metrics ke JSON
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    # Step 4: Save model
    save_model(model)
    print("Model terbaik disimpan ke 'best_model.joblib'")
