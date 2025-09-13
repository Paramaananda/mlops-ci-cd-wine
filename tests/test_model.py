from src.model import train_model, save_model, load_model
from src.preprocessing import load_and_preprocess
import os

def test_train_and_save_model(tmp_path):
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    save_path = tmp_path / "model.joblib"
    save_model(model, path=save_path)
    assert os.path.exists(save_path)

def test_load_model(tmp_path):
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    model = train_model(X_train, y_train)
    save_path = tmp_path / "model.joblib"
    save_model(model, path=save_path)
    loaded_model = load_model(path=save_path)
    assert loaded_model.predict(X_test[:1]).shape == (1,)
