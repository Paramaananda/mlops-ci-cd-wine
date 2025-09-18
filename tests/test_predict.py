from src.model import train_model, save_model
from src.preprocessing import load_and_preprocess
from src.predict import predict_new
import joblib
import numpy as np

def test_predict_new():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    model = train_model(X_train, y_train)
    save_model(model)
    joblib.dump(scaler, "scaler.joblib")  

    sample = X_test[0]
    pred = predict_new(sample)
    assert isinstance(pred, np.ndarray)
    assert pred[0] in [0, 1]
