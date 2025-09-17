import numpy as np
import pandas as pd
import joblib

def predict_new(sample, model_path="best_model.joblib", scaler_path="scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if hasattr(scaler, "feature_names_in_"):
        sample = pd.DataFrame([sample], columns=scaler.feature_names_in_)
    else:
        sample = np.array(sample).reshape(1, -1)

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)
    return pred
