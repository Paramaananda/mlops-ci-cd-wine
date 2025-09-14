from flask import Flask, request, jsonify
from src.model import load_model
import numpy as np
import joblib

app = Flask(__name__)

MODEL_PATH = "best_model.joblib"
model = load_model(MODEL_PATH)

try:
    scaler = joblib.load("scaler.joblib")
except:
    scaler = None

@app.route("/")
def home():
    return jsonify({"message": "ML Model API is running 🚀"}), 200


@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_json = request.get_json()
        data = input_json.get("data")

        if not data:
            return jsonify({"error": "Data input tidak ditemukan"}), 400

        arr = np.array(data).reshape(1, -1)

        if scaler:
            arr = scaler.transform(arr)

        prediction = model.predict(arr)
        pred_class = int(prediction[0])

        return jsonify({"prediction": pred_class}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
