"""
train.py
--------

Script to:
- load the Olivetti faces dataset
- create a 70/30 train/test split
- train a DecisionTreeClassifier
- save the trained model to ``saved_model.pth``
"""

from mlops_major.model_utils import (
    MODEL_PATH,
    evaluate_model,
    save_model,
    train_model,
)
from mlops_major.preprocess import get_train_test_split


def main():
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.3,
        random_state=42,
    )
    model = train_model(
        X_train,
        y_train,
        random_state=42,
    )
    save_model(model, MODEL_PATH)

    # Optional: print accuracy here as well for quick feedback
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Training completed. Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()


