from src.preprocessing import load_and_preprocess
from src.model import train_model, save_model
from src.evaluate import evaluate_model
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import subprocess
import os

MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
METRICS_PATH = "metrics.json"

if __name__ == "__main__":
    # ==============================
    # Setup MLflow local tracking
    # ==============================
    mlflow.set_tracking_uri("file:./mlruns") 
    experiment_name = "wine-quality"
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Hitung jumlah run yang sudah ada
    runs = client.search_runs(experiment.experiment_id)
    run_version = len(runs) + 1
    run_name = f"wine_quality_run_v{run_version}"

    # ==============================
    # Preprocessing & Training
    # ==============================
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    model = train_model(X_train, y_train)

    # ==============================
    # Evaluasi model
    # ==============================
    metrics = evaluate_model(model, X_test, y_test)
    print("📊 Evaluation Metrics:", metrics)

    # Simpan metrics JSON untuk DVC
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    # Simpan model & scaler
    save_model(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model disimpan ke '{MODEL_PATH}'")
    print(f"✅ Scaler disimpan ke '{SCALER_PATH}'")

    # ==============================
    # Logging ke MLflow
    # ==============================
    input_example = np.array([X_test[0]])

    with mlflow.start_run(run_name=run_name):
        # ==== PARAMS: data info ====
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("class_balance", dict(zip(*np.unique(y_train, return_counts=True))))

        # ==== PARAMS: model hyperparameters ====
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)
        if hasattr(model, "random_state"):
            mlflow.log_param("random_state", model.random_state)

        mlflow.log_param("model_type", type(model).__name__)

        # ==== METRICS ====
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # ==== ARTIFACTS ====
        # 1. Simpan model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # 2. Simpan metrics.json
        mlflow.log_artifact(METRICS_PATH)

        # 3. Confusion matrix plot
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 4. requirements.txt (jika ada)
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt")

        # ==== VERSION CONTROL INFO ====
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
            mlflow.log_param("git_commit", commit_hash)
        except Exception as e:
            print(f"⚠️ Gagal mengambil git commit hash: {e}")

    print(f"🚀 Run berhasil dicatat di MLflow dengan nama: {run_name}")
