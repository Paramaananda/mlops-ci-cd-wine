from src.model import train_model, save_model, load_model
from src.preprocessing import load_and_preprocess, StandardScaler
from src.predict import predict_new
import numpy as np

def test_predict_new():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    model = train_model(X_train, y_train)
    save_model(model)

    sample = X_test[0]
    pred = predict_new(sample)
    assert isinstance(pred[0], (int, np.integer))
