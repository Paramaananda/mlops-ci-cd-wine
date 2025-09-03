from src.model import train_model

def test_model_training():
    import numpy as np
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    model = train_model(X, y)
    assert model is not None
