from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path("saved_model.pth")


def train_model(X_train, y_train, **model_kwargs) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier on the provided data.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    model_kwargs : dict
        Extra keyword arguments passed to DecisionTreeClassifier.
    """
    clf = DecisionTreeClassifier(**model_kwargs)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test) -> float:
    """
    Compute accuracy of a trained model on the test set.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def save_model(model, path: Path = MODEL_PATH) -> None:
    """
    Persist the trained model to disk using joblib.
    """
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH):
    """
    Load a persisted model from disk.
    """
    return joblib.load(path)


__all__ = [
    "MODEL_PATH",
    "train_model",
    "evaluate_model",
    "save_model",
    "load_model",
]


