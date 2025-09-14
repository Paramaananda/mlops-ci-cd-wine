from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "best_model.joblib"

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)

def load_model(path=MODEL_PATH):
    return joblib.load(path)
