from src.model import load_model
import numpy as np

def predict(data):
    model = load_model()
    data = np.array(data).reshape(1, -1)
    return model.predict(data)
