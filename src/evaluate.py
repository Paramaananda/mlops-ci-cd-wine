from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
    }
    return metrics
